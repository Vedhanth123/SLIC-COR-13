@echo off
echo Running HDFC Chart Generator...
cd /d "e:\Ver6-COR-13-HDFC"
call .\env\Scripts\activate.bat
python generate_charts.py
echo.
echo Charts have been generated in the "charts" directory.
echo You can now browse all the charts in Windows Explorer.
echo.
pause
explorer "e:\Ver6-COR-13-HDFC\charts"
