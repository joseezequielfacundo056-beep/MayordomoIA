@echo off
setlocal enabledelayedexpansion
title Deploy - Mayordomo
cd /d "%~dp0"

echo.
echo ==============================================================
echo   DEPLOY DEL MAYORDOMO A RENDER (via GitHub)
echo ==============================================================
echo.
echo   Render no acepta subir un zip directo: necesita un repo de
echo   GitHub conectado. Este script hace la parte de GitHub por
echo   vos. Conectar ese repo en Render es un paso manual (una sola
echo   vez, dos clicks) que te explico al final.
echo.

REM --- Busca Git ---
where git >nul 2>&1
if errorlevel 1 (
    echo ==============================================================
    echo   No se encontro Git instalado en esta computadora.
    echo   Instalalo desde https://git-scm.com/downloads
    echo   Durante la instalacion podes dejar todas las opciones por
    echo   defecto. Despues de instalarlo, volve a hacer doble clic
    echo   en este archivo.
    echo ==============================================================
    echo.
    pause
    exit /b 1
)

REM --- Repo local ---
if not exist ".git" (
    echo Inicializando repositorio local...
    git init -q
    git branch -M main
)

REM --- Remoto de GitHub ---
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo.
    echo ==============================================================
    echo   Todavia no hay un repo de GitHub conectado.
    echo   1. Andate a https://github.com/new
    echo   2. Creale un nombre, dejalo PRIVADO si preferis, y NO
    echo      tildes "Add a README" ^(que quede vacio^)
    echo   3. Copia la URL que te da GitHub, algo como:
    echo      https://github.com/tu-usuario/tu-repo.git
    echo ==============================================================
    echo.
    start "" https://github.com/new
    set /p REPO_URL="Pega aca esa URL y apreta Enter: "
    if "!REPO_URL!"=="" (
        echo No pusiste ninguna URL. Volve a correr el script cuando la tengas.
        pause
        exit /b 1
    )
    git remote add origin "!REPO_URL!"
)

echo.
echo Preparando los archivos para subir...
if not exist ".env" (
    echo   ^(nota: no hay .env local, eso esta bien, esa key va directo
    echo    en Render y no se sube nunca al repo^)
)

git add -A
git commit -m "deploy %date% %time%" >nul 2>&1

echo Subiendo a GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo ==============================================================
    echo   Hubo un problema subiendo a GitHub. Lo mas comun es que
    echo   Windows te pida loguearte: se deberia abrir una ventana de
    echo   GitHub para iniciar sesion, completala y volve a correr
    echo   este archivo.
    echo ==============================================================
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================================================
echo   Listo, tu codigo ya esta en GitHub.
echo.
echo   Paso final ^(manual, una sola vez^):
echo   1. Entra a https://dashboard.render.com
echo   2. "New +" -^> "Web Service" -^> elegi ese repo
echo   3. Build command:  pip install -r requirements.txt
echo   4. Start command:  gunicorn server:app
echo   5. En "Environment", agrega GEMINI_API_KEY con tu key real
echo   6. Create Web Service
echo.
echo   Las proximas veces, con solo correr este mismo .bat de nuevo
echo   despues de cambiar algo, Render va a redeployar solo.
echo ==============================================================
echo.
start "" https://dashboard.render.com
pause
