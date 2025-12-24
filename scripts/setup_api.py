"""
Установка зависимостей для API
"""
import subprocess
import sys

def install_requirements():
    requirements = [
        'fastapi==0.104.1',
        'uvicorn[standard]==0.24.0',
        'python-multipart==0.0.6',
        'jinja2==3.1.2',
        'pillow==10.1.0',
        'opencv-python-headless==4.8.1',
        'ultralytics==8.0.0',
        'python-dotenv==1.0.0',
        'aiofiles==23.2.1'
    ]

    print("📦 Устанавливаю зависимости для API...")
    for package in requirements:
        print(f"  → {package}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

    print("✅ Все зависимости установлены!")

if __name__ == "__main__":
    install_requirements()
