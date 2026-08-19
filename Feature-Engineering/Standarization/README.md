# High-Performance Data Standardization (Python + C++)

Hey! this is a project I built to learn how to speed up machine learning data preprocessing. It takes a Python list of numbers and standardizes it (Z-score normalization) using a super-fast C++ backend. Instead of running slow Python for loops to calculate the mean and variance, it passes the data over to C++ via pybind11 and does the heavy lifting instantly. 
## Features

- Super Fast: Compiled with -O3 optimization flags so the loops run at bare-metal speed.
- Smart Memory Handling: Uses C++ references (const &) to pass the data without making slow, accidental copies in your computer's **RAM**.
- Easy to Use: Once compiled, you can literal run import my_Standarization directly inside a normal Python script.

## What You Need Make sure you have these installed on your Windows machine before running the build:

1. g++ / MinGW-w64 (Your C++ compiler, added to your environment variables).
2. Python 3.11 (The Windows Microsoft Store version).
3. pybind11 library. If you don't have it yet, get it via your terminal:
    
```
    pip install pybind11
```
    
## How to Compile (Windows PowerShell)[using WINDOWS for this project.]
Because the Microsoft Store version of Python locks down its system folders, standard setup.py scripts get confused. To bypass this, open PowerShell in your project folder and run this command to compile your module but keep in mind use you own folder path structure:

```
g++ -O3 -shared -std=c++11 -I"C:\Users\Abheet\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\pybind11\include* -I*C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.**496**.0_x64__qbz5n2kfra8p0\Include* cmain.cpp -o my_Standarization.pyd -L*C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.**496**.0_x64__qbz5n2kfra8p0\libs" -lpython311 -static -static-libgcc -static-libstdc++
```
## What do those flags actually do?

- `-O3`: Tells the compiler to violently optimize the math loops for speed.
- `-o` my_Standarization.pyd: Tells it to output a .pyd file, which is just a special Windows **DLL** file that Python can read as a library.
- `-static` : A crucial workaround that fixes a nasty Windows compiler bug related to clock libraries.