#!/usr/bin/env python
"""
Запуск API сервера
"""
from main import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Запуск ADAS API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)