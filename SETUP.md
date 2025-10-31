# Student Performance Analysis & Prediction System

A comprehensive data science application that uses machine learning to predict student academic performance based on attendance, study hours, and previous exam scores.

## Features

- **Performance Prediction**: ML-powered predictions using Linear Regression
- **Risk Assessment**: Automatic classification of students into risk categories
- **Data Visualization**: Interactive charts showing correlations and distributions
- **Student Records**: Database storage for tracking predictions over time
- **Recommendations**: Personalized suggestions for student improvement
- **Model Analytics**: View model performance metrics (R² score, RMSE, coefficients)

## Technology Stack

### Backend
- **Flask** - Python web framework
- **scikit-learn** - Machine learning library
- **pandas & numpy** - Data processing
- **matplotlib & seaborn** - Data visualization
- **Supabase** - PostgreSQL database

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with modern design
- **JavaScript (Vanilla)** - Interactive functionality

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**

   Your `.env` file should already contain Supabase credentials:
   ```
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**

   Open your browser and navigate to: `http://localhost:5000`

## Usage Guide

### 1. Making Predictions

- Navigate to the **Prediction** tab
- Enter student details:
  - Student Name
  - Attendance Percentage (0-100)
  - Study Hours per Day (0-24)
  - Previous Exam Score (0-100)
- Click "Predict Performance"
- View the predicted score, risk level, and recommendations

### 2. Analytics & Visualization

- Navigate to the **Analytics** tab
- Choose from three visualization types:
  - **Correlation Matrix**: See relationships between factors
  - **Scatter Plots**: View individual factor impacts
  - **Distribution**: Analyze data distributions

### 3. Model Information

- Navigate to the **Model Info** tab
- Click "Load Model Information"
- View:
  - R² Score (model accuracy)
  - RMSE (prediction error)
  - Model coefficients
  - Prediction equation

### 4. Student Records

- Navigate to the **Records** tab
- Click "Load Recent Records"
- View all past predictions with timestamps

## Project Structure

```
project/
├── app.py                      # Flask backend application
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # Main HTML template
├── static/
│   ├── css/
│   │   └── styles.css         # Styling
│   └── js/
│       └── app.js             # Frontend JavaScript
├── .env                        # Environment variables
└── SETUP.md                    # This file
```

## Machine Learning Model

The system uses **Linear Regression** to predict final exam scores based on:

- **Attendance**: Weight varies based on training data
- **Study Hours**: Direct correlation with performance
- **Previous Score**: Strong indicator of future performance

### Model Performance Metrics

- **R² Score**: Measures prediction accuracy (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)

### Risk Classification

- **Low Risk**: Predicted score ≥ 75
- **Medium Risk**: Predicted score 60-74
- **High Risk**: Predicted score < 60

## Database Schema

### student_records Table

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| student_name | text | Student name |
| attendance | numeric | Attendance % (0-100) |
| study_hours | numeric | Study hours per day |
| previous_score | numeric | Previous exam score |
| predicted_score | numeric | ML predicted score |
| risk_level | text | Risk assessment |
| created_at | timestamptz | Record timestamp |

## API Endpoints

- `GET /` - Main application page
- `POST /api/predict` - Make performance prediction
- `GET /api/model-info` - Get model metrics
- `POST /api/visualize` - Generate data visualizations
- `GET /api/students` - Get student records

## Future Enhancements

1. Mobile app development (Android/iOS)
2. Real-time performance tracking
3. Integration with educational platforms (Google Classroom, Moodle)
4. Advanced AI models (Neural Networks, Random Forest)
5. Personalized learning resource recommendations
6. Automated email/SMS alerts for at-risk students
7. Export reports to PDF/Excel
8. Multi-user access with role-based permissions
9. Progress tracking over time
10. Calendar integration

## Troubleshooting

### Python Dependencies Issues
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Port Already in Use
```bash
# Change port in app.py
app.run(debug=True, port=5001)
```

### Database Connection Issues
- Verify `.env` file contains correct Supabase credentials
- Check internet connection
- Ensure Supabase project is active

## Developer Information

**Project**: Student Performance Analysis & Prediction System
**Developer**: Ansal (22010203006)
**Institution**: Atal Bihari Vajpayee Govt. Institute of Engineering & Technology
**Guide**: Er. Anurag Sharma
**Academic Year**: 2025

## License

This project is developed for academic purposes as part of B.Tech Computer Science Engineering program.
