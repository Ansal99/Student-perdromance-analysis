from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
from dotenv import load_dotenv
import openai

# Compatibility: some openai package versions or local name collisions may not expose
# the `openai.error` module. Provide fallback exception aliases so our except
# clauses won't fail with "module 'openai' has no attribute 'error'".
try:
    OpenAIAuthenticationError = openai.error.AuthenticationError
    OpenAIRateLimitError = openai.error.RateLimitError
    OpenAIAPIError = openai.error.APIError
except Exception:
    class OpenAIAuthenticationError(Exception):
        pass

    class OpenAIRateLimitError(Exception):
        pass

    class OpenAIAPIError(Exception):
        pass

# Ensure the installed `supabase` package is imported instead of any local `supabase/` dir
import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
_removed = False
if _project_root in sys.path:
    try:
        sys.path.remove(_project_root)
        _removed = True
    except ValueError:
        _removed = False

from supabase import create_client, Client

# restore sys.path so application relative imports still work
if _removed:
    sys.path.insert(0, _project_root)
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

app = Flask(__name__)
CORS(app)

supabase_url = os.getenv('VITE_SUPABASE_URL')
supabase_key = os.getenv('VITE_SUPABASE_ANON_KEY')

# Initialize OpenAI API key and print status for debugging
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    print("Warning: OpenAI API key is not set. Please add OPENAI_API_KEY to your .env file.")
else:
    print("OpenAI API key is configured.")

# Detect common placeholder API key values to avoid hitting the API with an invalid key.
try:
    if openai.api_key and isinstance(openai.api_key, str) and 'your' in openai.api_key.lower():
        print("Warning: OPENAI_API_KEY appears to be a placeholder. Please replace it with a valid key from https://platform.openai.com/account/api-keys")
        # Unset to prevent accidental attempts with an obvious placeholder
        openai.api_key = None
except Exception:
    # If something odd happens, continue without raising during import
    openai.api_key = openai.api_key if getattr(openai, 'api_key', None) else None

# Initialize Supabase client if credentials are present
supabase: Client | None = None
if supabase_url and supabase_key:
    try:
        # Create Supabase client without proxy configuration
        supabase = create_client(
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )
    except Exception as e:
        print(f"Warning: failed to initialize Supabase client: {e}")
        print("If you don't need Supabase functionality, you can ignore this warning.")
        supabase = None

model = None
feature_names = ['attendance', 'study_hours', 'previous_score']

def train_initial_model():
    global model
    sample_data = pd.DataFrame({
        'attendance': [85, 90, 75, 95, 80, 70, 88, 92, 78, 85, 93, 68, 82, 87, 91, 73, 89, 94, 76, 84],
        'study_hours': [5, 6, 4, 7, 5, 3, 6, 7, 4, 5, 6, 3, 5, 6, 7, 4, 6, 7, 4, 5],
        'previous_score': [75, 80, 70, 85, 72, 65, 78, 82, 68, 74, 83, 62, 71, 77, 81, 67, 79, 84, 69, 73],
        'final_score': [78, 83, 72, 88, 75, 67, 80, 85, 70, 76, 86, 64, 73, 79, 84, 69, 81, 87, 71, 75]
    })

    X = sample_data[feature_names]
    y = sample_data['final_score']

    model = LinearRegression()
    model.fit(X, y)

    return model

train_initial_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        attendance = float(data.get('attendance', 0))
        study_hours = float(data.get('study_hours', 0))
        previous_score = float(data.get('previous_score', 0))
        student_name = data.get('student_name', 'Anonymous')

        input_data = np.array([[attendance, study_hours, previous_score]])
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))

        risk_level = 'Low Risk' if prediction >= 75 else 'Medium Risk' if prediction >= 60 else 'High Risk'
        risk_color = 'success' if prediction >= 75 else 'warning' if prediction >= 60 else 'danger'

        recommendations = []
        if attendance < 75:
            recommendations.append('Improve attendance to at least 75%')
        if study_hours < 4:
            recommendations.append('Increase daily study hours to minimum 4 hours')
        if previous_score < 70:
            recommendations.append('Focus on strengthening fundamentals')
        if prediction < 75:
            recommendations.append('Consider additional tutoring or study groups')

        if not recommendations:
            recommendations.append('Keep up the excellent work!')

        if supabase:
            try:
                supabase.table('student_records').insert({
                    'student_name': student_name,
                    'attendance': attendance,
                    'study_hours': study_hours,
                    'previous_score': previous_score,
                    'predicted_score': round(prediction, 2),
                    'risk_level': risk_level
                }).execute()
            except Exception as e:
                print(f"Database error: {e}")

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        sample_data = pd.DataFrame({
            'attendance': [85, 90, 75, 95, 80, 70, 88, 92, 78, 85, 93, 68, 82, 87, 91, 73, 89, 94, 76, 84],
            'study_hours': [5, 6, 4, 7, 5, 3, 6, 7, 4, 5, 6, 3, 5, 6, 7, 4, 6, 7, 4, 5],
            'previous_score': [75, 80, 70, 85, 72, 65, 78, 82, 68, 74, 83, 62, 71, 77, 81, 67, 79, 84, 69, 73],
            'final_score': [78, 83, 72, 88, 75, 67, 80, 85, 70, 76, 86, 64, 73, 79, 84, 69, 81, 87, 71, 75]
        })

        X = sample_data[feature_names]
        y = sample_data['final_score']

        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)

        coefficients = {
            'attendance': round(model.coef_[0], 4),
            'study_hours': round(model.coef_[1], 4),
            'previous_score': round(model.coef_[2], 4),
            'intercept': round(model.intercept_, 4)
        }

        return jsonify({
            'success': True,
            'r2_score': round(r2, 4),
            'rmse': round(rmse, 4),
            'coefficients': coefficients
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/visualize', methods=['POST'])
def visualize():
    try:
        data = request.json
        chart_type = data.get('chart_type', 'correlation')

        sample_data = pd.DataFrame({
            'attendance': [85, 90, 75, 95, 80, 70, 88, 92, 78, 85, 93, 68, 82, 87, 91, 73, 89, 94, 76, 84],
            'study_hours': [5, 6, 4, 7, 5, 3, 6, 7, 4, 5, 6, 3, 5, 6, 7, 4, 6, 7, 4, 5],
            'previous_score': [75, 80, 70, 85, 72, 65, 78, 82, 68, 74, 83, 62, 71, 77, 81, 67, 79, 84, 69, 73],
            'final_score': [78, 83, 72, 88, 75, 67, 80, 85, 70, 76, 86, 64, 73, 79, 84, 69, 81, 87, 71, 75]
        })

        plt.figure(figsize=(10, 6))

        if chart_type == 'correlation':
            correlation_matrix = sample_data[feature_names + ['final_score']].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix - Student Performance Factors', fontsize=14, fontweight='bold')

        elif chart_type == 'scatter':
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].scatter(sample_data['attendance'], sample_data['final_score'], alpha=0.6, color='#3b82f6')
            axes[0].set_xlabel('Attendance (%)')
            axes[0].set_ylabel('Final Score')
            axes[0].set_title('Attendance vs Final Score')
            axes[0].grid(True, alpha=0.3)

            axes[1].scatter(sample_data['study_hours'], sample_data['final_score'], alpha=0.6, color='#10b981')
            axes[1].set_xlabel('Study Hours (per day)')
            axes[1].set_ylabel('Final Score')
            axes[1].set_title('Study Hours vs Final Score')
            axes[1].grid(True, alpha=0.3)

            axes[2].scatter(sample_data['previous_score'], sample_data['final_score'], alpha=0.6, color='#f59e0b')
            axes[2].set_xlabel('Previous Score')
            axes[2].set_ylabel('Final Score')
            axes[2].set_title('Previous Score vs Final Score')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

        elif chart_type == 'distribution':
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            axes[0, 0].hist(sample_data['attendance'], bins=10, color='#3b82f6', alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Attendance (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Attendance Distribution')
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].hist(sample_data['study_hours'], bins=8, color='#10b981', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Study Hours (per day)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Study Hours Distribution')
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].hist(sample_data['previous_score'], bins=10, color='#f59e0b', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Previous Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Previous Score Distribution')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].hist(sample_data['final_score'], bins=10, color='#ef4444', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Final Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Final Score Distribution')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{image_base64}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        if not supabase:
            return jsonify({'success': False, 'error': 'Database not configured'}), 400

        response = supabase.table('student_records').select('*').order('created_at', desc=True).limit(50).execute()

        return jsonify({
            'success': True,
            'students': response.data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        print("Chat endpoint called")  # Debug log
        
        # Check if OpenAI API key is configured
        if not openai.api_key:
            print("Error: OpenAI API key is missing")  # Debug log
            return jsonify({
                'success': False,
                'error': 'OpenAI API key is not configured. Please add OPENAI_API_KEY to your .env file.'
            }), 400

        data = request.json
        print(f"Received data: {data}")  # Debug log
        
        if not data:
            print("Error: No request data")  # Debug log
            return jsonify({
                'success': False,
                'error': 'No data provided in the request'
            }), 400

        user_message = data.get('message', '')
        print(f"User message: {user_message}")  # Debug log
        
        if not user_message:
            print("Error: Empty message")  # Debug log
            return jsonify({
                'success': False,
                'error': 'No message provided in the request'
            }), 400
        
        # Create context about the student performance system
        system_context = """You are an AI academic advisor assistant in a student performance analysis system. 
        You can help with:
        1. Interpreting student performance predictions
        2. Providing study tips and strategies
        3. Explaining the importance of attendance and study hours
        4. Suggesting ways to improve academic performance
        5. Understanding the relationships between different performance factors
        Be concise, supportive, and provide actionable advice."""
        
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_message}
        ]
        
        try:
            print("Attempting to call OpenAI API...")  # Debug log

            ai_response = None

            # Support both new and old openai-python SDKs.
            # Prefer new interface (openai.OpenAI) when available (openai>=1.0.0).
            # Fallback to old openai.ChatCompletion for older SDKs.
            client = None
            if hasattr(openai, 'OpenAI'):
                # new client available
                try:
                    # prefer constructing with env-based config
                    try:
                        client = openai.OpenAI()
                    except Exception:
                        # try passing api_key explicitly
                        client = openai.OpenAI(api_key=openai.api_key) if openai.api_key else openai.OpenAI()
                except Exception as e:
                    print(f"Could not initialize new OpenAI client: {e}")
                    client = None

            if client:
                # call new client API
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7
                )

                # Try common response shapes for new client
                ai_response = None
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        ai_response = choice.message.content
                    elif isinstance(choice, dict):
                        ai_response = (choice.get('message') or {}).get('content') or choice.get('text') or choice.get('delta', {}).get('content')
                    else:
                        ai_response = str(choice)
            else:
                # Fall back to old SDK usage
                if hasattr(openai, 'ChatCompletion'):
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=300,
                        temperature=0.7
                    )
                    # old response shape
                    ai_response = None
                    try:
                        if hasattr(response, 'choices') and len(response.choices) > 0:
                            # response.choices[0].message.content or response.choices[0].text
                            choice = response.choices[0]
                            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                                ai_response = choice.message.content
                            elif isinstance(choice, dict):
                                ai_response = (choice.get('message') or {}).get('content') or choice.get('text')
                            else:
                                ai_response = str(choice)
                    except Exception:
                        ai_response = None
                else:
                    raise RuntimeError('OpenAI client not available in installed openai package')

            if not ai_response:
                raise RuntimeError('Failed to retrieve assistant response from OpenAI API')

            print(f"Got AI response: {ai_response[:100]}...")  # Debug log truncated response

            return jsonify({
                'success': True,
                'response': ai_response
            })
        except OpenAIAuthenticationError as e:
            print(f"Authentication Error: {str(e)}")  # Debug log
            return jsonify({
                'success': False,
                'error': 'Invalid OpenAI API key. Please check your OPENAI_API_KEY in the .env file.'
            }), 401
        except OpenAIRateLimitError as e:
            print(f"Rate Limit Error: {str(e)}")  # Debug log
            return jsonify({
                'success': False,
                'error': 'OpenAI API rate limit exceeded. Please try again later.'
            }), 429
        except OpenAIAPIError as e:
            print(f"API Error: {str(e)}")  # Debug log
            return jsonify({
                'success': False,
                'error': f'OpenAI API error: {str(e)}'
            }), 500
        except Exception as e:
            print(f"Unexpected error in OpenAI call: {str(e)}")  # Debug log
            return jsonify({
                'success': False,
                'error': f'An unexpected error occurred while processing your request: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Chat error: {str(e)}")  # Log the error for debugging
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

@app.route('/api/health', methods=['GET'])
def health():
    """Simple health endpoint to check OpenAI key and Supabase initialization."""
    try:
        openai_ok = bool(openai.api_key)
    except Exception:
        openai_ok = False

    supabase_ok = supabase is not None
    return jsonify({
        'success': True,
        'openai_configured': openai_ok,
        'supabase_initialized': supabase_ok
    })
