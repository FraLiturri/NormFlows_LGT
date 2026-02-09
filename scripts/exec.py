import subprocess
import os
import sys

# Parametri
START, END, DELTA = 1.0, 7.0, 0.1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_lr(val):
    if val <= 2: return "1e-3"
    if val <= 3: return "1e-4"
    if val <= 4.4: return "7.5e-5"
    if val <= 6: return "1e-6"
    return "0.5e-6"

def get_epoch(val):
    return "20" if val == 1.0 else "100"

val = START
with open(os.path.join(SCRIPT_DIR, "report.log"), "a") as log_file:
    while val <= END + 1e-9:  # Aggiunto piccolo epsilon per precisione float
        lr = get_lr(val)
        epoch = get_epoch(val)
        
        print(f"Eseguendo val={val:.1f}, epoch={epoch}, lr={lr}...")
        
        # Esegue python usando lo stesso interprete corrente
        cmd = [sys.executable, "-u", "u1_forward.py", f"{val:.1f}", epoch, lr, str(DELTA)]
        
        result = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=log_file)
        
        if result.returncode != 0:
            print(f"Errore rilevato con input {val:.1f}. Stop.")
            sys.exit(1)
            
        val += DELTA

print("Completato!")