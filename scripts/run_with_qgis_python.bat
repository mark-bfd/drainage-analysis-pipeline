@echo off
REM Run Python scripts using QGIS's Python environment (has GDAL, numpy, etc.)
REM Usage: run_with_qgis_python.bat script.py [args...]

set QGIS_ROOT=C:\Program Files\QGIS 3.40.15
set QGIS_PYTHON="%QGIS_ROOT%\apps\Python312\python.exe"

if not exist %QGIS_PYTHON% (
    echo ERROR: QGIS Python not found at %QGIS_PYTHON%
    echo Please install QGIS 3.40.15 or update the path in this script.
    exit /b 1
)

REM Set environment for GDAL, GRASS, and PROJ
set PATH=%QGIS_ROOT%\bin;%QGIS_ROOT%\apps\gdal\bin;%QGIS_ROOT%\apps\grass\grass84\bin;%QGIS_ROOT%\apps\grass\grass84\lib;%PATH%
set GDAL_DATA=%QGIS_ROOT%\apps\gdal\share\gdal
set PROJ_LIB=%QGIS_ROOT%\share\proj
set PYTHONPATH=%QGIS_ROOT%\apps\qgis-ltr\python;%QGIS_ROOT%\apps\Python312\lib\site-packages
set GISBASE=%QGIS_ROOT%\apps\grass\grass84

%QGIS_PYTHON% %*
