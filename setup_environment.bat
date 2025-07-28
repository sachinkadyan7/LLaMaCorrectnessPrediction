@REM Install Miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o ..\miniconda.exe
start /wait "" .\miniconda /S
del ./miniconda.exe
%WINDIR%\System32\cmd.exe "/K" %LOCALAPPDATA%\miniconda3\Scripts\activate.bat %LOCALAPPDATA%\miniconda3

@REM Install Dependencies
conda create -p ..\llama-env python==3.9 -y
conda activate llama-env
pip3 install -r requirements.txt

@REM Set Environment Variables
set HF_HOME=..\hf_data
set HF_TOKEN=""

@REM End of Script