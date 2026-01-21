# Test Experiment: Class 0 vs Class 1

## Overview

A working example experiment has been created with **real visualizations** from your continuous learning test run.

## Experiment Details

- **Experiment ID**: `class0_vs_class1_20260113_171544`
- **Name**: Class 0 vs Class 1 - Test Experiment
- **Status**: Completed
- **Progress**: 10 cycles completed (out of 336 total)
- **Current Accuracy**: 95.2%
- **QC Pass Rate**: 9/10 (90%)

## Available Visualizations

The experiment includes **5 real figures** from your test run:

1. **Universal Classification** (835 KB)
   - UMAP/PCA/t-SNE dimensionality reduction plots
   - Shows class separation in feature space

2. **GA Progress Realtime** (558 KB)
   - Genetic algorithm optimization progress
   - Shows convergence over generations

3. **Feature Importance** (333 KB)
   - Feature ranking and importance scores
   - Shows which audio features matter most

4. **Accuracy Plot** (19 KB)
   - Training and validation accuracy over epochs
   - Shows model learning progress

5. **Loss Plot** (17 KB)
   - Training and validation loss over epochs
   - Shows optimization effectiveness

## How to View

### 1. Access the Dashboard

Open your browser and go to:
```
http://localhost:5001/continuous
```

### 2. Find the Experiment

Look for the card titled:
**"Class 0 vs Class 1 - Test Experiment"**

It should show:
- Status badge: **COMPLETED** (blue)
- Progress: 10 / 336 cycles
- Accuracy: 95.2%
- QC Pass Rate: 90%

### 3. Click to View Details

Click anywhere on the experiment card to open the details modal.

### 4. Navigate the Tabs

The modal has 5 tabs:

#### **Overview Tab** (default)
- Shows configuration (substances, intervals, etc.)
- Shows current status with progress bar
- Displays experiment metadata

#### **Statistics Tab**
- Metric cards: QC passes/fails, accuracy stats
- Accuracy trend table showing cycle-by-cycle progress

#### **Cycles Tab**
- Scrollable table with all 10 completed cycles
- Shows QC status, separation scores, accuracy per cycle
- Training time and completion timestamps

#### **Visualizations Tab** ⭐
**This is what you want to see!**

When you click this tab:
1. It will load all 5 visualizations
2. Shows them in a responsive grid (3-4 columns)
3. Each thumbnail is clickable

**To view full-size**:
- Click any thumbnail
- Opens full-screen modal with high-resolution image
- Click X or outside to close
- Navigate back to gallery for other images

#### **Alerts Tab**
(Only appears if there are alerts - not present in this test)

## File Locations

The visualizations are stored at:
```
/Users/bernd/python/XOR/models/continuous/class0_vs_class1_20260113_171544/plots/
```

Original files copied from:
```
/Volumes/CIH/mora/xor/experiments/continuous_test_config_20260113_171558/plots/
```

## Testing the API Directly

You can also test the API endpoints directly:

### Get Experiment Details
```bash
curl http://localhost:5001/api/continuous/experiments/class0_vs_class1_20260113_171544 | python3 -m json.tool
```

### Get Statistics
```bash
curl http://localhost:5001/api/continuous/experiments/class0_vs_class1_20260113_171544/stats | python3 -m json.tool
```

### Get Visualizations List
```bash
curl http://localhost:5001/api/continuous/experiments/class0_vs_class1_20260113_171544/visualizations | python3 -m json.tool
```

### View a Specific Image
Open in browser:
```
http://localhost:5001/models/continuous/class0_vs_class1_20260113_171544/plots/universal_classification_20260113_183612.png
```

## Troubleshooting

### Can't See Visualizations Tab
- Make sure you clicked on the experiment card
- Modal should open with tabs at the top
- Click the "Visualizations" tab (4th tab)

### Images Not Loading
Check server logs:
```bash
tail -f /Users/bernd/python/XOR/web_server.log
```

Verify files exist:
```bash
ls -lh models/continuous/class0_vs_class1_20260113_171544/plots/
```

### Server Not Running
Restart it:
```bash
lsof -ti:5001 | xargs kill -9
/Users/bernd/Library/r-miniconda/envs/xorProject/bin/python web/app.py --port 5001 --host 0.0.0.0 > web_server.log 2>&1 &
```

## Creating More Test Experiments

To add visualizations from other experiments:

1. **Create directory**:
   ```bash
   mkdir -p models/continuous/your_experiment_id/plots
   ```

2. **Copy your PNG/JPG files**:
   ```bash
   cp /path/to/your/figures/*.png models/continuous/your_experiment_id/plots/
   ```

3. **Create database entry** (adapt the Python script from earlier)

4. **Refresh dashboard** - new experiment will appear

## Features Demonstrated

This test experiment demonstrates all the new UI features:

✓ **Tabbed interface** - Organized content in 5 sections
✓ **Scrollable modal** - Handles large amounts of data
✓ **Visualization gallery** - Grid layout with thumbnails
✓ **Click to enlarge** - Full-screen image viewer
✓ **Auto-discovery** - Automatically found 5 PNG files
✓ **Smart captions** - Generated from filenames
✓ **Lazy loading** - Images load only when tab clicked
✓ **Responsive design** - Works on different screen sizes
✓ **Statistics cards** - Visual metrics display
✓ **Cycle history** - Detailed table with all data

Enjoy exploring your experiment results!
