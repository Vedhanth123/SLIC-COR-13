@echo off
REM filepath: e:\Ver6-COR-13-HDFC\export_recommendations.bat
echo HDFC Recommendations Export Tool
echo -------------------------------

cd /d "e:\Ver6-COR-13-HDFC"
call .\env\Scripts\activate.bat

python export_recommendations.py

echo.
echo Press any key to exit...
pause > nul
