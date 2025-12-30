# Week 1, Session 1.1, Topic 4: Environment Setup
## Duration: 20 minutes

---

## Learning Objectives
By the end of this topic, students will be able to:
1. Install Python and Anaconda on their system
2. Launch and use Jupyter Notebooks
3. Install essential AI/ML libraries (NumPy, Pandas, Matplotlib)
4. Use Google Colab as a cloud alternative
5. Verify their environment is working correctly
6. Know where to get help when issues arise

---

## Pre-Session Preparation (Send to Students 24 Hours Before)

**Email Template:**

```
Subject: Action Required: Set Up Your AI Development Environment

Hi everyone!

To make the most of tomorrow's session, please try to complete these steps beforehand. 
Don't worry if you run into issues - we'll troubleshoot together during class!

OPTION 1: Local Installation (Recommended)
1. Download Anaconda: https://www.anaconda.com/download
2. Choose your operating system (Windows/Mac/Linux)
3. Download the installer (it's ~500MB)
4. Run the installer (accept defaults)
5. This will take 10-15 minutes

OPTION 2: Cloud-Based (Backup)
1. Create a Google account if you don't have one
2. Visit: https://colab.research.google.com
3. Click "New Notebook"
4. You're ready!

We'll walk through everything step-by-step in class, so don't stress if you 
encounter problems. See you tomorrow!

Best,
[Instructor Name]

P.S. Having both options available is ideal - local for daily work, 
Colab for when you need more power or are away from your computer.
```

---

## Lecture Script & Content

### Opening (2 minutes)

**Instructor Script:**

"Alright team! We've covered a lot of theory in the past hour:
- What AI and Machine Learning are
- The different types of ML
- How Deep Learning works and when to use it

Now it's time to get practical! You can't learn to code AI without actually coding, right?

**Today's Goal:**
By the end of this session, everyone will have a working environment where you can:
- Write and run Python code
- Use Jupyter Notebooks
- Import ML libraries
- Run your first AI code!

**Quick Survey:**
- Who already has Python installed? [Show hands]
- Who installed Anaconda before class? [Show hands]
- Who's never programmed before? [Show hands]

Perfect! No matter where you're starting from, we'll get everyone ready.

**Two Paths We'll Cover:**
1. **Local Setup** (Anaconda) - Run on your computer
2. **Cloud Setup** (Google Colab) - Run in your browser

I recommend having both! Local for daily work, Colab for when you need GPU power or are traveling.

Let's start with the local setup..."

---

### Part 1: Installing Python & Anaconda (8 minutes)

**Slide 1: Why Anaconda?**

**Content:**
```
Why Not Just Install Python?

Python Alone:
- Need to install packages one by one
- Managing dependencies is painful
- Different projects need different versions
- Environment conflicts

Anaconda:
✓ Python + 250+ data science packages
✓ Includes Jupyter Notebooks
✓ Package management (conda)
✓ Environment management
✓ Works on Windows, Mac, Linux
✓ One installation = Everything you need
```

**Instructor Script:**

"First question: Why Anaconda instead of just installing Python?

Think of it like this:
- **Just Python** = Buying a car with no accessories
- **Anaconda** = Buying a fully loaded car with GPS, premium sound, everything

Anaconda includes:
- Python itself
- Jupyter Notebooks (our primary tool)
- NumPy, Pandas, Matplotlib (our essential libraries)
- 250+ other useful packages
- Conda (package manager that prevents conflicts)

One installation, everything works together. That's why 90% of data scientists use it!

**Step-by-Step Installation:**

Let me share my screen and walk through this together. Follow along if you haven't installed yet.

---

**For Windows Users:**

```
Step 1: Download
→ Go to: https://www.anaconda.com/download
→ Click "Download" for Windows
→ Choose 64-bit installer (most computers)
→ File is ~500MB, takes 2-5 minutes to download

Step 2: Run Installer
→ Double-click the .exe file
→ Click "Next" through the screens
→ IMPORTANT: Check "Add Anaconda to PATH" (makes life easier)
→ Choose "Install for: Just Me" (recommended)
→ Installation takes 5-10 minutes

Step 3: Verify Installation
→ Open Command Prompt (search "cmd" in Start menu)
→ Type: conda --version
→ Should see: conda 23.x.x (some version number)
→ Type: python --version
→ Should see: Python 3.11.x or similar
```

**Common Windows Issues:**

```
Issue: "conda not recognized"
Solution: Need to add to PATH manually
1. Search "Environment Variables" in Start
2. Edit "Path" variable
3. Add: C:\Users\[YourName]\anaconda3\Scripts
4. Restart Command Prompt

Issue: "Permission denied"
Solution: Right-click installer → "Run as Administrator"
```

---

**For Mac Users:**

```
Step 1: Download
→ Go to: https://www.anaconda.com/download
→ Click "Download" for macOS
→ Choose the right chip:
  - M1/M2/M3 Mac: Apple Silicon
  - Older Mac: Intel
→ File is ~500MB

Step 2: Run Installer
→ Double-click the .pkg file
→ Click "Continue" through screens
→ Install in default location
→ Takes 5-10 minutes

Step 3: Verify Installation
→ Open Terminal (Cmd+Space, type "Terminal")
→ Type: conda --version
→ Should see: conda 23.x.x
→ Type: python --version
→ Should see: Python 3.11.x
```

**Common Mac Issues:**

```
Issue: Command not found
Solution: Need to initialize conda
→ Type: source ~/anaconda3/bin/activate
→ Then: conda init zsh (or bash)
→ Restart Terminal

Issue: Security warning
Solution: 
→ System Preferences → Security & Privacy
→ Click "Open Anyway" for Anaconda installer
```

---

**For Linux Users:**

```
Step 1: Download
→ wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
→ (Use latest version from anaconda.com)

Step 2: Install
→ bash Anaconda3-2024.02-1-Linux-x86_64.sh
→ Follow prompts, accept license
→ Install in default location: ~/anaconda3
→ Yes to "conda init"

Step 3: Verify
→ Close and reopen terminal
→ conda --version
→ python --version
```

---

**Live Installation Demo:**

[Instructor shares screen]

"Let me show you what a successful installation looks like...

[Open terminal/command prompt]

```bash
# Check conda
$ conda --version
conda 24.1.0

# Check Python
$ python --version
Python 3.11.7

# List installed packages
$ conda list
# Shows hundreds of packages!
```

If you see version numbers, you're good! If you see errors, don't panic - we'll help troubleshoot."

**Visual Aid Description:**
[Show side-by-side screenshots of successful installation on Windows and Mac. Highlight the commands and expected outputs.]

---

### Part 2: Jupyter Notebooks (5 minutes)

**Slide 2: Introduction to Jupyter Notebooks**

**Content:**
```
What is Jupyter Notebook?

Interactive coding environment:
- Write code in cells
- Run cells individually
- See output immediately
- Mix code, text, and visualizations
- Perfect for learning and experimentation

Why Data Scientists Love It:
✓ Immediate feedback
✓ Document your thinking
✓ Share work easily
✓ Visualize results inline
✓ Experiment without risk
```

**Instructor Script:**

"Now that you have Python installed, let's talk about HOW we'll write code.

We could use a basic text editor, but that's like using a typewriter when you could have a computer! Instead, we'll use **Jupyter Notebooks**.

**What Makes Jupyter Special:**

Think of it like a lab notebook:
- Traditional coding: Write entire program → Run → See all results
- Jupyter: Write small piece → Run → See result → Write next piece

**Perfect for Learning:**
- Try something → See if it works immediately
- Make mistake → Fix just that part
- Visualize data → Right there in the notebook
- Add notes → Explain your thinking

**Real Example:**

Traditional Python file (.py):
```python
# Have to run entire file at once
print("Step 1")
print("Step 2")
print("Step 3")
# If step 2 has error, must fix and re-run ALL steps
```

Jupyter Notebook:
```python
# Cell 1 - Run independently
print("Step 1")
[Run] ✓

# Cell 2 - Run independently
print("Step 2")
[Run] ✓

# Cell 3 - Run independently
print("Step 3")
[Run] ✓
```

If Cell 2 has an error, just fix and re-run that cell!

---

**Launching Jupyter Notebook:**

[Share screen]

**Method 1: Anaconda Navigator (Easy, Visual)**

```
Step 1: Open Anaconda Navigator
→ Windows: Search "Anaconda Navigator" in Start
→ Mac: Applications → Anaconda Navigator

Step 2: Launch Jupyter
→ Find "Jupyter Notebook" tile
→ Click "Launch"
→ Browser opens automatically
→ You'll see your file system
```

**Method 2: Command Line (Faster)**

```
Step 1: Open Terminal/Command Prompt

Step 2: Type command
$ jupyter notebook

Step 3: Browser opens automatically
→ Shows: http://localhost:8888/tree
→ You see your home directory
```

**Live Demo:**

[Instructor demonstrates launching Jupyter]

"Watch what happens when I run 'jupyter notebook'...

```bash
$ jupyter notebook
[I 10:30:21.456 NotebookApp] Serving notebooks from local directory: /Users/ali
[I 10:30:21.456 NotebookApp] Jupyter Notebook 6.5.4 is running at:
[I 10:30:21.456 NotebookApp] http://localhost:8888/?token=abc123...
```

See how my browser automatically opened? This is your Jupyter interface.

---

**Creating Your First Notebook:**

```
Step 1: Click "New" (top right)
Step 2: Select "Python 3"
Step 3: You now have a blank notebook!
Step 4: Rename it (click "Untitled" at top)
```

**Interface Tour:**

[Point out key elements]

```
Menu Bar:
- File: Save, download, etc.
- Edit: Cut/copy cells
- Cell: Run cells, cell type
- Kernel: Restart Python

Toolbar:
- ▶ Run cell
- ➕ Add cell
- ✂ Cut cell
- 📋 Copy cell

Cell:
- Click to edit
- Shift+Enter to run
- Code or Markdown
```

**Your First Code:**

Let's write and run something!

```python
# Cell 1: Basic Python
print("Hello, AI World!")
print("2 + 2 =", 2 + 2)
```

[Press Shift+Enter]

Output:
```
Hello, AI World!
2 + 2 = 4
```

**Try More:**

```python
# Cell 2: Variables
name = "Your Name"
course = "AI Engineering"
print(f"I'm {name}, learning {course}!")
```

**Markdown Cells:**

```markdown
# This is a Title
## This is a subtitle

You can write **bold** and *italic* text.

Create lists:
- Point 1
- Point 2
- Point 3
```

**Keyboard Shortcuts (Most Useful):**

```
Command Mode (press Esc):
- A: Add cell above
- B: Add cell below
- DD: Delete cell
- M: Change to Markdown
- Y: Change to Code

Edit Mode (press Enter):
- Shift+Enter: Run cell, move to next
- Ctrl+Enter: Run cell, stay in cell
- Alt+Enter: Run cell, insert below
```

This will become second nature quickly!"

**Visual Aid Description:**
[Show annotated screenshot of Jupyter interface with labels pointing to key areas. Include example of code cell and markdown cell side by side.]

---

### Part 3: Installing Essential Libraries (3 minutes)

**Slide 3: Essential Python Libraries for AI**

**Content:**
```
The AI/ML Python Stack:

NumPy: Numerical computing, arrays
Pandas: Data manipulation, DataFrames
Matplotlib: Data visualization, plots
Scikit-learn: Classical ML algorithms
PyTorch: Deep Learning (Week 3+)

Good News: Anaconda includes most of these!
```

**Instructor Script:**

"Good news! If you installed Anaconda, you already have most libraries we need!

Let's verify and install anything missing.

**Check What's Installed:**

In a Jupyter cell, run:

```python
# Check NumPy
import numpy as np
print(f"NumPy version: {np.__version__}")

# Check Pandas
import pandas as pd
print(f"Pandas version: {pd.__version__}")

# Check Matplotlib
import matplotlib.pyplot as plt
print(f"Matplotlib version: {matplotlib.__version__}")

# Check Scikit-learn
import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")
```

If all these work, you're set! ✓

**If Something's Missing:**

```python
# In terminal/command prompt
$ conda install numpy pandas matplotlib scikit-learn

# Or for a specific package
$ conda install pytorch

# Alternative: using pip
$ pip install package-name
```

**Installing PyTorch (We'll need this in Week 3):**

PyTorch needs special installation based on your system.

Visit: https://pytorch.org/get-started/locally/

Select:
- Your OS (Windows/Mac/Linux)
- Conda or Pip
- CPU or CUDA (GPU)

Copy the command shown, run it in terminal.

Example (CPU version):
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Don't worry about this now - we'll do it together in Week 3!

**Quick Test:**

Let's make sure everything works:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create simple data
x = np.array([1, 2, 3, 4, 5])
y = x ** 2

# Create DataFrame
df = pd.DataFrame({'x': x, 'y': y})
print(df)

# Plot
plt.plot(x, y)
plt.title("My First Plot!")
plt.show()
```

If you see a plot, everything is working! 🎉"

---

### Part 4: Google Colab (Cloud Alternative) (4 minutes)

**Slide 4: Google Colab**

**Content:**
```
Google Colab: Jupyter in the Cloud

Advantages:
✓ No installation needed
✓ Access from anywhere
✓ Free GPU access
✓ Preinstalled libraries
✓ Easy sharing
✓ Saves to Google Drive

When to Use:
- Need GPU for deep learning
- On a different computer
- Installation problems
- Quick experiments
- Collaborating with others
```

**Instructor Script:**

"Sometimes you can't or don't want to use your local computer:
- Traveling with tablet
- Installation issues
- Need more power
- Want to collaborate

That's where Google Colab shines!

**Getting Started with Colab:**

```
Step 1: Open browser
→ Go to: https://colab.research.google.com

Step 2: Sign in
→ Use Google account
→ Don't have one? Create free account

Step 3: Create notebook
→ Click "New Notebook"
→ Or: File → New Notebook
```

**Colab Interface:**

Looks almost identical to Jupyter!
- Same cell system
- Same keyboard shortcuts
- Same code works

**Key Differences:**

```
Colab:
- Runs on Google servers
- Files save to Google Drive
- Can enable GPU/TPU
- Already has libraries installed
- Free tier has time limits

Local Jupyter:
- Runs on your computer
- Files save locally
- Uses your CPU/GPU
- You install libraries
- No time limits
```

**Enabling GPU (For Deep Learning):**

```
Step 1: Runtime → Change runtime type
Step 2: Hardware accelerator: GPU
Step 3: Save

Now you have free GPU! 🚀
```

**Testing Colab:**

Run the same code:

```python
# Already installed in Colab!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"PyTorch: {torch.__version__}")

# Check if GPU available
if torch.cuda.is_available():
    print("✓ GPU is available!")
else:
    print("Using CPU (enable GPU in Runtime settings)")
```

**Uploading Files to Colab:**

```python
# Upload from computer
from google.colab import files
uploaded = files.upload()

# Or mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

**Downloading Files from Colab:**

```python
files.download('filename.csv')
```

**Best Practice:**

I recommend:
- **Primary**: Use local Jupyter for daily work
- **Backup**: Use Colab when you need GPU or are away from computer
- **Collaboration**: Share Colab links with teammates

All our course materials will work on both!"

**Visual Aid Description:**
[Show side-by-side comparison of Jupyter Notebook interface vs. Google Colab interface. Highlight similarities and differences.]

---

### Part 5: Verification & Troubleshooting (3 minutes)

**Slide 5: Environment Checklist**

**Content:**
```
✓ Verification Checklist:

□ Python 3.8+ installed
□ Can launch Jupyter Notebook
□ Can create and run cells
□ NumPy imports successfully
□ Pandas imports successfully
□ Matplotlib creates plots
□ Saved a test notebook
□ (Optional) Google Colab account created

If all checked: You're ready! 🎉
```

**Instructor Script:**

"Let's make sure everyone's environment is working!

**Complete This Verification Notebook:**

Create a new notebook called `environment_test.ipynb`:

```python
# Cell 1: Check Python version
import sys
print(f"Python version: {sys.version}")
assert sys.version_info >= (3, 8), "Need Python 3.8+"
print("✓ Python version OK")
```

```python
# Cell 2: Check NumPy
import numpy as np
print(f"NumPy version: {np.__version__}")
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {arr.mean()}")
print("✓ NumPy working")
```

```python
# Cell 3: Check Pandas
import pandas as pd
print(f"Pandas version: {pd.__version__}")
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
print(df)
print("✓ Pandas working")
```

```python
# Cell 4: Check Matplotlib
import matplotlib.pyplot as plt
print(f"Matplotlib version: {plt.matplotlib.__version__}")

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
plt.plot(x, y, 'ro-')
plt.title('Test Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
print("✓ Matplotlib working")
```

```python
# Cell 5: System info
import platform
print(f"Operating System: {platform.system()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print("\n✅ ALL CHECKS PASSED! Environment ready!")
```

**If Everything Works:**
You should see all the ✓ checkmarks and a plot!

**Common Issues & Fixes:**

```
Problem: "ModuleNotFoundError: No module named 'numpy'"
Solution: 
$ conda install numpy
Or restart Jupyter after installation

Problem: Plot doesn't show
Solution: Add this to first cell:
%matplotlib inline

Problem: Jupyter won't start
Solution:
1. Check if already running (close other instances)
2. Try: jupyter notebook --port 8889 (different port)
3. Reinstall: conda install jupyter

Problem: Permission errors
Solution:
- Windows: Run as Administrator
- Mac/Linux: Check folder permissions

Problem: Kernel keeps dying
Solution:
1. Restart kernel: Kernel → Restart
2. Update: conda update jupyter
3. Create new environment (advanced)
```

**Getting Help:**

```
During This Course:
- Ask in our Discord/Slack channel
- Office hours
- Email instructor: hi@bytebytego.com

Online Resources:
- Stack Overflow (search your error)
- Anaconda documentation
- Jupyter documentation
- Python Discord communities
```

**Environment Ready?**

Quick poll:
- ✅ Environment working? Give thumbs up
- ⚠️ Having issues? We'll help after session
- 🌥️ Using Colab instead? That's fine too!

Don't let technical issues stop you - we have solutions!"

---

### Closing & Looking Ahead (1 minute)

**Slide 6: You're Ready to Code!**

**Content:**
```
What We've Set Up:

✓ Python programming environment
✓ Jupyter Notebooks for interactive coding
✓ Essential libraries (NumPy, Pandas, Matplotlib)
✓ Cloud backup (Google Colab)

Next Session:
- Python refresher
- NumPy for AI
- Pandas for data
- Matplotlib for visualization

Then: Build your first ML model! 🚀
```

**Instructor Script:**

"Congratulations! You now have a complete AI development environment!

**What You Can Do Now:**
- Write and run Python code
- Work with data using Pandas
- Perform numerical computations with NumPy
- Create visualizations with Matplotlib
- Use Jupyter Notebooks like a pro

**Next Steps:**

After this break, we'll dive into Session 1.2:
- Python refresher (or intro if you're new)
- NumPy essentials for AI
- Pandas for data manipulation
- Matplotlib for visualization

By the end of today, you'll have written actual code and analyzed real data!

**Before We Break:**

Please do this now:
1. Create a notebook called `week1_practice.ipynb`
2. Run the verification code
3. Save it
4. Raise hand if you have issues

We have a 10-minute break. If your environment isn't working, stay and we'll troubleshoot together. Otherwise, stretch, grab coffee, and come back ready to code!

Questions about the setup? [Pause for questions]

Great! See you in 10 minutes for Python and AI libraries! 🎉"

---

## Visual Aids & Slides Summary

**Slide 1: Why Anaconda?**
- Comparison table: Python vs. Anaconda
- List of included packages
- Installation badges for Windows/Mac/Linux

**Slide 2: Introduction to Jupyter Notebooks**
- Screenshot of Jupyter interface (annotated)
- Cell types comparison (Code vs. Markdown)
- Keyboard shortcuts cheat sheet

**Slide 3: Essential Libraries**
- The AI/ML Python stack diagram
- Library icons and purposes
- Installation commands

**Slide 4: Google Colab**
- Colab interface screenshot
- Advantages list
- Jupyter vs. Colab comparison

**Slide 5: Verification Checklist**
- Interactive checklist
- Common errors and solutions table
- Support resources

**Slide 6: Ready to Code**
- Summary of achievements
- Preview of next session
- Motivational image

---

## Hands-On Activity: Environment Setup Challenge (During Session)

**Activity: Everyone Runs This Together**

Create a new notebook and run this code:

```python
# AI Environment Setup Challenge
# Goal: Verify all components working

print("="*50)
print("AI ENGINEERING BOOTCAMP - ENVIRONMENT TEST")
print("="*50)

# Test 1: Imports
print("\n[Test 1] Importing libraries...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✓ All libraries imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")

# Test 2: NumPy computation
print("\n[Test 2] NumPy computation...")
arr = np.random.randn(5, 5)
result = arr.mean()
print(f"✓ Created 5x5 array, mean = {result:.4f}")

# Test 3: Pandas DataFrame
print("\n[Test 3] Pandas DataFrame...")
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Score': [95, 87, 92, 88]
})
print("✓ Created DataFrame:")
print(df)

# Test 4: Visualization
print("\n[Test 4] Creating visualization...")
fig, ax = plt.subplots(figsize=(8, 5))
df.plot(x='Name', y='Score', kind='bar', ax=ax, color='skyblue')
ax.set_title('Student Scores', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
plt.tight_layout()
plt.show()
print("✓ Plot created successfully")

# Success!
print("\n" + "="*50)
print("🎉 CONGRATULATIONS! Your environment is ready!")
print("="*50)
```

**Expected Output:**
- All ✓ checkmarks
- DataFrame displayed
- Bar chart shown
- Success message

---

## Troubleshooting Guide (Reference Document)

**Issue 1: Anaconda Won't Install**

```
Symptom: Installer fails or hangs

Solutions:
1. Check disk space (need ~5GB free)
2. Disable antivirus temporarily
3. Download fresh installer (file may be corrupted)
4. Try Miniconda instead (lighter version)
   → https://docs.conda.io/en/latest/miniconda.html
```

**Issue 2: Jupyter Won't Launch**

```
Symptom: Command not found or browser doesn't open

Solutions:
1. Restart terminal/command prompt
2. Verify installation: conda list jupyter
3. Reinstall: conda install jupyter
4. Try: python -m notebook
5. Check if port 8888 is blocked: jupyter notebook --port 8889
```

**Issue 3: Import Errors**

```
Symptom: "ModuleNotFoundError" when importing

Solutions:
1. Install missing package: conda install package-name
2. Verify kernel: Kernel → Change Kernel → Python 3
3. Restart kernel: Kernel → Restart
4. Check Python path: import sys; print(sys.executable)
```

**Issue 4: Kernel Dies/Crashes**

```
Symptom: "Kernel has died, restarting"

Solutions:
1. Close other applications (free up RAM)
2. Restart Jupyter completely
3. Update conda: conda update --all
4. Create new environment (advanced):
   conda create -n ai_course python=3.11
   conda activate ai_course
```

**Issue 5: Can't Save Notebooks**

```
Symptom: Save fails or permission denied

Solutions:
1. Check folder permissions
2. Save to different location
3. Run Jupyter from your home directory
4. Windows: Don't run from Program Files
```

**Issue 6: Internet Connection Issues (for Colab)**

```
Symptom: Colab won't load or keeps disconnecting

Solutions:
1. Check internet connection
2. Try different browser (Chrome works best)
3. Clear browser cache
4. Disable browser extensions
5. Use Colab in incognito mode
```

---

## Resources for Students

**Official Documentation:**
- Anaconda: https://docs.anaconda.com/
- Jupyter: https://jupyter-notebook.readthedocs.io/
- Google Colab: https://colab.research.google.com/notebooks/intro.ipynb

**Video Tutorials:**
- Installing Anaconda (Windows/Mac)
- Jupyter Notebook tutorial for beginners
- Google Colab crash course

**Cheat Sheets:**
- Jupyter Notebook keyboard shortcuts
- Conda commands cheat sheet
- Markdown syntax guide

**Community Support:**
- Course Discord/Slack channel
- r/learnpython subreddit
- Stack Overflow
- Python Discord server

---

## Instructor Notes

**Pacing:**
- This is hands-on, not just lecture
- Pause frequently to let students try
- Have TAs/helpers ready to assist
- Don't rush - environment issues are frustrating

**Key Messages:**
1. Technical setup is normal and everyone struggles
2. Multiple paths to success (local + cloud)
3. Don't get discouraged by installation issues
4. We'll help you get working

**Common Pitfalls:**
- Students feel behind if installation fails
- Some won't try beforehand (plan for this)
- Version conflicts (especially on older systems)
- Firewall/antivirus blocking installations

**Engagement:**
- Make it collaborative ("Help your neighbor")
- Celebrate when things work
- Normalize troubleshooting
- Share your own past installation struggles

**Backup Plans:**
- Have Colab ready as fallback
- Pre-recorded installation videos
- Written step-by-step guides
- Office hours for persistent issues

**Testing Before Class:**
- Test installation on Windows/Mac/Linux
- Verify all links work
- Have offline installers ready
- Prepare common error screenshots

---

## Post-Session Follow-Up

**Email to Students:**

```
Subject: Session 1.1 Complete - Environment Setup Resources

Great job today! You made it through setup - that's often the hardest part!

Resources:
1. Verification notebook: [link]
2. Installation troubleshooting guide: [link]
3. Jupyter keyboard shortcuts: [link]
4. Next session prep: Review Python basics (optional)

Still Having Issues?
- Post in Discord/Slack #technical-help
- Office hours: [time]
- Email: hi@bytebytego.com

See you in Session 1.2 where we'll start coding with Python, NumPy, and Pandas!

Best,
[Instructor]
```

---

## Assessment Check

**At the end of this topic, students should:**
- ✅ Have working Python environment (local or cloud)
- ✅ Be able to launch Jupyter Notebook
- ✅ Create and run cells
- ✅ Import NumPy, Pandas, Matplotlib
- ✅ Save and reopen notebooks
- ✅ Know where to get help

**Quick Verification:**
Students should successfully run the verification notebook and see all ✓ marks.

---

## Preparation Checklist for Instructor

- [ ] Send pre-class setup email 24-48 hours before
- [ ] Test installations on multiple operating systems
- [ ] Prepare troubleshooting documentation
- [ ] Have offline installers ready
- [ ] Set up help Discord/Slack channel
- [ ] Recruit TAs to help with setup
- [ ] Prepare verification notebook file
- [ ] Test Google Colab access from classroom
- [ ] Have backup internet connection
- [ ] Prepare screen sharing for demos

---

## Transition to Session 1.2

**Instructor Script:**

"Excellent work everyone! You now have your AI development environment ready to go.

[Quick poll]
- Who has local Jupyter working? 🙋
- Who is using Colab? 🙋
- Who still needs help? (Stay after for assistance)

After our break, we're jumping into **Session 1.2: Python for AI**

We'll cover:
- Python refresher (quick!)
- NumPy: The foundation of numerical computing
- Pandas: Your data manipulation powerhouse
- Matplotlib: Creating beautiful visualizations

And we'll end with an actual hands-on activity analyzing real data!

Take a 10-minute break. When you come back, have your Jupyter Notebook open and ready!

Need help? Now's the time - I'll be here during break! ☕"

---

**Next Session:** Session 1.2 - Python for AI (90 minutes)
