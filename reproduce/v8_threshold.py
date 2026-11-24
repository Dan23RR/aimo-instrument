"""Section 6: the two parameters of our entry V8 (threshold 98 characters, direction long = robust).

Usage, from the repository root:  python archive/dati.py   then   python archive/aggrega.py   then   python reproduce/v8_threshold.py
Same computation as ricalcola_v8.py in the working folder (archive/ in the public repository; standard library only);
expected output: lower median 98, balanced accuracy 0.65245 against 0.34755 on the 335 public pairs and 0.63057
against 0.36943 on the 283 distinct ones.
"""
import os
import runpy

from common import ROOT

runpy.run_path(os.path.join(ROOT, "ricalcola_v8.py"), run_name="__main__")
