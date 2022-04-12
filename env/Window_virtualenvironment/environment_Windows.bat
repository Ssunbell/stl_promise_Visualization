@REM set root=C:\anaconda3

@REM %root%\Scripts\activate.bat %root% 

call python -m pip install --upgrade pip
call conda create -n semi_proj python=3.9 -y
call conda env list
call conda activate semi_proj
call pip install streamlit
call pip install -r requirements.txt
call cd ..
call streamlit run promise.py

pause