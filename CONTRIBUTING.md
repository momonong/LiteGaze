#  小組開發協作規範 (Git Guidelines)

為了讓大家合作愉快且 Repo 不會爆炸，請各位組員遵守以下簡單的開發原則。

##  分支地圖
* **`main`**: 最終成果分支。除非是最後要交作業或 Release，否則**請勿動它**。
* **`group`**: **我們小組的主分支**。所有的開發成果最後都要合併到這裡。
* **開發分支**: 每個功能或測試請從 `group` 分支出去，完成後再合回 `group`。

---

##  開發流程
1.  **開工前先更新**: 確保你的本地 `group` 是最新的。
    ```bash
    git checkout group
    git pull origin group
    ```
2.  **建立新分支**: 根據任務類型命名（參考下方規範）。
    ```bash
    git checkout -b feat/your-feature-name
    ```
3.  **開發與提交**: 正常的 git commit。
4.  **發送 Pull Request (PR)**: 
    * 開發完成後推送到 GitHub。
    * **重要：** 請務必選擇合併到 **`group`** 分支，而不是 `main`。

---

##  分支命名規範 (Branch Naming)
為了避免衝突（特別是不要再用 `group/xxx` 這種命名），請統一使用以下前綴：

| 前綴 | 使用時機 | 範例 |
| :--- | :--- | :--- |
| `feat/` | 開發新功能 | `feat/login-page` |
| `fix/` | 修復 Bug | `fix/header-logo` |
| `test/` | 寫測試、實驗功能 | `test/api-performance` |
| `refactor/` | 重構程式碼（不改功能） | `refactor/clean-code` |
| `docs/` | 改文件 (README 等) | `docs/update-install-guide` |

>  **注意**：請一律使用**小寫**，並用**連字號 `-`** 分隔單字。

---

##  Commit 訊息小提醒
雖然不強制，但建議寫清楚這筆改動做了什麼，例如：
* `feat: 新增登入功能介面`
* `fix: 修正首頁按鈕點擊沒反應的問題`

---

大家加油！有問題隨時在群組討論。
