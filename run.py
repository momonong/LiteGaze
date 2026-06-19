# IMPORTANT: Start with `python -X utf8 run.py` to prevent UnicodeEncodeError
# crashes on Windows CP950 when library code prints non-Big5 characters.
import webbrowser
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'

# Import and build Flask application factory
from web_app import create_app
app = create_app()

if __name__ == '__main__':
    url = 'http://localhost:8080/'
    print('=' * 48)
    print('  文件座標擷取工具  —  Flask Backend (Refactored)')
    print('=' * 48)
    print(f'  網址  : {url}')
    print(f'  資料  : {DATA_DIR}')
    print(f'  API   : http://localhost:8080/api/sessions')
    print(f'  認知負荷 : http://localhost:8080/api/cognitive/health')
    print(f'  停止  : Ctrl + C')
    print('=' * 48)
    webbrowser.open(url)
    app.run(host='localhost', port=8080, debug=False)
