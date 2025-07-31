# How to run on Compute Canada

Currently, the Nibi is still quite flaky, so Graham is recommended.

---

## To log in to Graham:

```bash
ssh user_name@graham.computecanada.ca
```

## To log into Nibi (when stabilized):

```bash
ssh user_name@nibi.sharcnet.ca
```

---

Once inside, and you have your Python scripts or bash available locally, copy your files into the environment.

_Refer to the GitHub repository for these files_: [Github Repo](https://github.com/nehaniv/Evolution-of-Complexity-in-Mealey-Automata/tree/main)

---

## To copy individual files to Graham:

```bash
scp /path/to/MealyEvolutionStrategy.sh /path/to/ES-Automata-Fogel.py user_name@graham.computecanada.ca:
```

---

## If you want to copy a whole folder:

```bash
scp -r /path/to/folder_name user_name@graham.computecanada.ca:
```

---

## 📂 Important files:

- `MealyEvolutionStrategy.sh`
- `ES-Automata-Fogel.py`
- `requirements.txt`

---

## 🐍 Before running the scripts in the environment

Activate your virtual environments with these commands:

```bash
python3 -m venv automata-env
source automata-env/bin/activate  # or 'automata-env\Scripts\activate' on Windows

# Install dependencies
pip install -r requirements.txt
```

---

If there are errors like matplotlib not available, I find the following command to be helpful:

```bash
module load python3
module load scipy-stack
```

---

Make sure you have your GAP up and running!

### To check run:

```bash
gap
```

---

### If not, follow these commands:

**_(FOLLOW THEM AS THEY ARE, DO NOT TRY TO SHORTCUT)_**

1. Download `gap-4.14.0.tar.gz` from [https://www.gap-system.org/install/linux](https://www.gap-system.org/install/linux)  
2. In home directory in alliance run:  
   ```bash
   mkdir -p ~/gap && cd ~/gap
   ```
3. Copy archive to `~/gap` with `scp`
4. Unpack with tar:  
   ```bash
   tar -xzf gap-4.14.0.tar.gz
   ```
5. 
   ```bash
   cd gap-4.14.0 && run ./configure && make
   ```
6. 
   ```bash
   cd to pkg and run ~/gap/gap-4.14.0/bin/BuildPackages.sh
   ```
7. 
   ```bash
   Check ~/gap/gap-4.14.0/pkg/log/fail.log
   ```
8. Add gap to path with nano ~/.bashrc by adding line:  
   ```bash
   export PATH="$HOME/gap:$PATH"
   ```
9. Download `sgpdec-master.zip` from:  
   [https://github.com/gap-packages/sgpdec](https://github.com/gap-packages/sgpdec)
10. Copy archive to `~/gap/gap-4.14.0/pkg` with scp and unzip
11. Add SgpDec for automatic loading:  
    ```bash
    echo 'LoadPackage("SgpDec");' >> ~/.gap
    ```
12. Check if GAP starts from the home directory with the `"gap"` command.

---

## ✅ After everything is up and running, try running your script!

```bash
# Remember to chmod +x MealyEvolutionStrategy.sh
python3 ES-Automata-Fogel.py \
  --population_size 5 \
  --offspring_size 10 \
  --num_states 15 \
  --runs 10 \
  --generations 1000 \
  --fitness MultiTraversal \
  --env_variant SimpleHardestEnvironment \
  --self_loop_init True \
  --execution Parallel
```

_**Edit the parameters according to your needs.**_

---

## To run it as job batches in Compute Canada, do:

```bash
# Remember to chmod +x MealyEvolutionStrategy.sh
sbatch python3 ES-Automata-Fogel.py \
  --population_size 5 \
  --offspring_size 10 \
  --num_states 15 \
  --runs 10 \
  --generations 1000 \
  --fitness MultiTraversal \
  --env_variant SimpleHardestEnvironment \
  --self_loop_init True \
  --execution Parallel
```

---

## To check for your batch status, do:

```bash
squeue -u $USER  # you can see the status of your script
```

---

## Afterwards, to retrieve the evolution report into your local:

```bash
scp nghipdo@graham.computecanada.ca:~/evolution_report_20250731_114842.pdf ~/Downloads/
```

---
