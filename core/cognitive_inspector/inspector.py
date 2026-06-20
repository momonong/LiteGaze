import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from wordfreq import zipf_frequency

@dataclass
class GazeFixation:
    word: str
    index: int
    duration_ms: float
    timestamp_ms: float
    confidence: str

class CognitiveInspector:
    def __init__(self, sample_rate_hz: int = 8):
        self.tick_ms = 1000 // sample_rate_hz  # ~120ms per tick

    def group_fixations(self, gaze_history: List[Dict[str, Any]]) -> List[GazeFixation]:
        """
        將連續的相同單字 index 點聚合為單一注視事件 (Fixation)。
        """
        if not gaze_history:
            return []

        # Calculate estimated tick duration dynamically from timestamps
        timestamps = sorted([hit.get("timestamp_ms", 0) for hit in gaze_history if "timestamp_ms" in hit])
        deltas = []
        for i in range(1, len(timestamps)):
            d = timestamps[i] - timestamps[i-1]
            if d > 0:
                deltas.append(d)

        if deltas:
            deltas.sort()
            median_delta = deltas[len(deltas) // 2]
            estimated_tick = min(max(median_delta, 50.0), 2000.0)
        else:
            estimated_tick = self.tick_ms

        group_threshold = max(350.0, estimated_tick * 1.5)

        fixations: List[GazeFixation] = []
        current_group = []

        for hit in gaze_history:
            idx = hit.get("index", -1)
            word = hit.get("word", "")
            if idx == -1 or not word:
                continue

            if not current_group:
                current_group.append(hit)
            else:
                last_hit = current_group[-1]
                # 若為相同單字 index 且時間間隔小於 group_threshold，視為同一注視
                if last_hit.get("index") == idx and (hit.get("timestamp_ms", 0) - last_hit.get("timestamp_ms", 0)) < group_threshold:
                    current_group.append(hit)
                else:
                    # 聚合上一個組
                    fixations.append(self._aggregate_group(current_group, estimated_tick))
                    current_group = [hit]

        if current_group:
            fixations.append(self._aggregate_group(current_group, estimated_tick))

        return fixations

    def _aggregate_group(self, group: List[Dict[str, Any]], tick_ms: float) -> GazeFixation:
        first = group[0]
        # 信心度排序：high > medium > low
        rank = {"high": 2, "medium": 1, "low": 0}
        best_conf = "low"
        for hit in group:
            conf = hit.get("confidence", "low")
            if rank.get(conf, 0) > rank.get(best_conf, 0):
                best_conf = conf

        count = len(group)
        duration = count * tick_ms

        return GazeFixation(
            word=first.get("word", ""),
            index=first.get("index", -1),
            duration_ms=duration,
            timestamp_ms=first.get("timestamp_ms", 0),
            confidence=best_conf
        )

    def analyze(self, gaze_history: List[Dict[str, Any]], lang: str = "en") -> Dict[str, Any]:
        """
        對使用者的眼動序列與語言複雜度進行全面性認知分析。
        """
        fixations = self.group_fixations(gaze_history)
        total_fixations = len(fixations)

        if total_fixations == 0:
            return self._empty_result()

        # 1. 基礎指標計算
        total_dwell_time_ms = sum(f.duration_ms for f in fixations)
        avg_fixation_duration = total_dwell_time_ms / total_fixations

        # 2. 回看次數 (Regression Count)
        regressions = 0
        last_index = -1
        for f in fixations:
            if last_index != -1 and f.index < last_index:
                regressions += 1
            last_index = f.index

        regression_rate = regressions / total_fixations if total_fixations > 0 else 0.0

        # 3. 重讀次數 (Reread Count)
        rereads = 0
        visited_indices = set()
        last_index = -1
        for f in fixations:
            if f.index != last_index:
                if f.index in visited_indices:
                    rereads += 1
                visited_indices.add(f.index)
            last_index = f.index

        # 4. 閱讀速度 (Words Per Minute)
        dwell_time_min = total_dwell_time_ms / 60000.0
        unique_words_count = len(visited_indices)
        wpm = unique_words_count / dwell_time_min if dwell_time_min > 0 else 0.0
        wpm = min(wpm, 600.0) # 合理上限

        # ── 評估維度 1：閱讀能力 (Reading Ability) ──────────────────────────────
        # WPM: 230+ = 100分, 70- = 30分
        wpm_score = 30.0 + (min(max(wpm, 70.0), 230.0) - 70.0) / 160.0 * 70.0
        # Regression Rate: <=0.05 = 100分, >=0.25 = 30分
        reg_score = 100.0 - (min(max(regression_rate, 0.05), 0.25) - 0.05) / 0.20 * 70.0
        # Fixation Duration: <=220ms = 100分, >=450ms = 30分
        fix_score = 100.0 - (min(max(avg_fixation_duration, 220.0), 450.0) - 220.0) / 230.0 * 70.0

        reading_ability = 0.4 * wpm_score + 0.3 * reg_score + 0.3 * fix_score

        # ── 評估維度 2：英語水準 (English Proficiency) ───────────────────────────
        # 交叉比對「受阻單字」(長注視或回看) 與詞頻 (Zipf frequency)
        struggled_words = []
        for f in fixations:
            # 長注視或觸發回看（此單字低於前一個，或被重讀）
            is_struggle = f.duration_ms > 350 or f.index in visited_indices
            if is_struggle:
                struggled_words.append(f.word)

        if struggled_words:
            zipfs = []
            for w in struggled_words:
                clean_w = w.strip(".,;:?!'\"()").lower()
                if clean_w:
                    zipfs.append(zipf_frequency(clean_w, lang))
            avg_struggle_zipf = sum(zipfs) / len(zipfs) if zipfs else 4.0
            
            # 若受阻單字平均詞頻低（代表只卡在罕見生字），英文水準高
            # Zipf <= 3.0 -> 95分, Zipf >= 5.5 -> 40分
            proficiency_score = 95.0 - (min(max(avg_struggle_zipf, 3.0), 5.5) - 3.0) / 2.5 * 55.0
        else:
            avg_struggle_zipf = 0.0
            proficiency_score = 88.0  # 閱讀極度流暢的基礎高分

        # ── 評估維度 3：注意力與疲勞度 (Attention & Fatigue) ────────────────────
        # 疲勞度評估：比較前半段與後半段的注視時長
        if total_fixations >= 4:
            mid = total_fixations // 2
            first_half = fixations[:mid]
            second_half = fixations[mid:]
            duration_first = sum(f.duration_ms for f in first_half) / len(first_half)
            duration_second = sum(f.duration_ms for f in second_half) / len(second_half)
            fatigue_ratio = duration_second / duration_first if duration_first > 0 else 1.0
        else:
            fatigue_ratio = 1.0

        if fatigue_ratio > 1.20:
            fatigue_level = "high"
            fatigue_label = "明顯疲勞 (注意力衰退，注視時間拉長)"
        elif fatigue_ratio > 1.05:
            fatigue_level = "medium"
            fatigue_label = "輕微疲勞"
        else:
            fatigue_level = "low"
            fatigue_label = "精神集中 / 狀態穩定"

        # 注意力穩定度：由 Fixation Count (命中次數) 分佈判定
        # 命中次數高於 unique 單字數，代表反覆跳視
        attention_index = max(10, min(100, int(100 - (regressions * 5 + rereads * 3))))

        # ── 綜合負荷 (Overall Load) ──────────────────────────────────────────
        cognitive_load = (avg_fixation_duration / 450.0 * 60) + (regression_rate * 40)
        cognitive_load = min(100.0, max(10.0, cognitive_load))

        return {
            "summary": {
                "total_fixations": total_fixations,
                "total_dwell_time_ms": int(total_dwell_time_ms),
                "avg_fixation_duration_ms": round(avg_fixation_duration, 1),
                "regression_count": regressions,
                "regression_rate": round(regression_rate, 3),
                "reread_count": rereads,
                "words_per_minute": round(wpm, 1),
                "unique_words_read": unique_words_count
            },
            "user_profile": {
                "reading_ability_score": round(reading_ability, 1),
                "reading_ability_level": self._score_to_level(reading_ability),
                "english_proficiency_score": round(proficiency_score, 1),
                "english_proficiency_level": self._score_to_level(proficiency_score),
                "avg_struggle_word_frequency": round(avg_struggle_zipf, 2),
                "cognitive_load_index": round(cognitive_load, 1),
                "fatigue_level": fatigue_level,
                "fatigue_label": fatigue_label,
                "fatigue_ratio": round(fatigue_ratio, 3),
                "attention_index": attention_index
            }
        }

    def _score_to_level(self, score: float) -> str:
        if score >= 85: return "Advanced / 優異"
        if score >= 70: return "Intermediate-High / 中上"
        if score >= 55: return "Intermediate / 中等"
        return "Novice / 待加強"

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_fixations": 0,
                "total_dwell_time_ms": 0,
                "avg_fixation_duration_ms": 0,
                "regression_count": 0,
                "regression_rate": 0,
                "reread_count": 0,
                "words_per_minute": 0,
                "unique_words_read": 0
            },
            "user_profile": {
                "reading_ability_score": 0,
                "reading_ability_level": "None",
                "english_proficiency_score": 0,
                "english_proficiency_level": "None",
                "avg_struggle_word_frequency": 0,
                "cognitive_load_index": 0,
                "fatigue_level": "none",
                "fatigue_label": "無資料",
                "fatigue_ratio": 1.0,
                "attention_index": 0
            }
        }
