@echo off
echo HDFC Analysis Dashboard Launcher
echo ------------------------------
echo 1. Run Full Dashboard (All tabs)
echo 2. Run Simple Dashboard with Custom Recommendations (One category at a time)
echo ------------------------------
choice /C 12 /M "Select an option:"

if errorlevel 2 goto SIMPLE
if errorlevel 1 goto FULL

:FULL
echo Starting HDFC Full Analysis Dashboard...
cd /d "e:\Ver6-COR-13-HDFC"
call .\env\Scripts\activate.bat
streamlit run streamlit_dashboard.py
goto END

:SIMPLE
echo Starting HDFC Simple Analysis Dashboard with Custom Recommendations...
cd /d "e:\Ver6-COR-13-HDFC"
call .\env\Scripts\activate.bat
streamlit run streamlit_dashboard_simple.py
goto END

:END
pause
