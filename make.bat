@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=./doc
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)


if "%1" == "deep_autograde" (
	%SPHINXBUILD% -b latex -t deepdive -D master_doc=source/deep_autograde %SOURCEDIR% %BUILDDIR%\deep_autograde
	cd %BUILDDIR%\deep_autograde
	make || exit /b 1
	cd %~dp0

	if not exist %BUILDDIR%\html mkdir %BUILDDIR%\html
	if not exist %BUILDDIR%\html\articles mkdir %BUILDDIR%\html\articles
	copy /Y %BUILDDIR%\deep_autograde\tinycio.pdf %BUILDDIR%\html\articles\deep_autograde.pdf

	goto end
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end


:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%


:end
popd
