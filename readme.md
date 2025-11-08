Project Structure :-

student_clustering_project/
│
├── cluster_analysis.py       # Main clustering script
├── elbow_plot.png            # Saved Elbow Method plot (auto-generated)
├── students_survey.csv       # Dataset file (input)
├── venv/                     # Virtual environment (optional)
└── README.md                 # Documentation

⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/LEYTHAN-X/student_clustering_project.git
cd student_clustering_project

2. Create and Activate Virtual Environment (Recommended)
python -m venv venv
# Activate it:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
pip install pandas scikit-learn matplotlib

📊 How It Works
Step 1: Data Preparation

Loads students_survey.csv

Cleans invalid rows and converts all numeric fields properly

Encodes categorical features (e.g., Internet speed, device type)

Scales all features using StandardScaler

Step 2: Finding the Optimal Number of Clusters

Uses the Elbow Method to plot inertia vs k

Saves the plot as elbow_plot.png

You can visually inspect the plot to determine the best k (default is 4)

Step 3: K-Means Clustering

Applies K-Means with the chosen k

Evaluates cluster quality using Silhouette Score

A score > 0.5 indicates good separation between clusters

Step 4: Cluster Analysis

Adds cluster labels back to the original dataset

Displays average values for numeric features and most common values for categorical ones

Helps interpret cluster personas

▶️ Running the Script

Simply execute:

python cluster_analysis.py


Output:

Elbow plot → elbow_plot.png

Silhouette score in terminal

Cluster summary table (mean and mode analysis)

📈 Example Output
Terminal:
Elbow plot saved as 'elbow_plot.png'. Please inspect it to confirm k=4.

K-Means complete with k=4.
Silhouette Score: 0.653
(This score is > 0.5, which meets your project's goal! ✅)

--- Cluster Analysis (Persona Profiles) ---
           StudyHoursPerDay  TechSkill  Motivation   Age  Device  Internet ...
Cluster
0                    2.75        3.5         4.0   20.1  Laptop       Fast
1                    1.20        2.1         3.5   19.8  Mobile     Average
...

--- Next Steps ---
Use the analysis table above to create your qualitative personas.

🧠 Insights You Can Derive

Identify clusters of students who:

Study less but have high tech skills

Are highly motivated but lack good internet

Depend heavily on mobile devices for learning

Build personalized learning plans or engagement strategies.

🪄 Customization

You can modify the clustering parameters:

labels = run_kmeans(scaled_data, k=4)  # Change k to test different cluster counts


Or replace the dataset with your own:

students_survey.csv  →  your_dataset.csv


Just ensure the new CSV follows similar column structure.

🧾 Requirements

Python 3.8+

pandas

scikit-learn

matplotlib

📤 Output Files
File	Description
elbow_plot.png	Elbow Method visualization
cluster_analysis.py	Main clustering script
students_survey.csv	Input dataset
students_clustered.csv (optional future output)	Clustered dataset