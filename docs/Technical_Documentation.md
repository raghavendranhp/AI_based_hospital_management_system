# Technical Documentation

## Architecture
The system follows a modular architecture:
1. **Data Layer (`src/data_generator.py`)**: Responsible for simulating real-world hospital operational data.
2. **Machine Learning Layer (`src/ml_models.py`, `src/train_models.py`)**: Handles the offline training pipeline, serialized model generation (`models/`), and inference logic.
3. **Intelligence Layer (`src/llm_agent.py`)**: Interacts with the Groq API (llama-3.1-8b-instant) to extract narrative insights from statistical summaries.
4. **Presentation Layer (`app.py`, `src/ui_components.py`)**: Streamlit application providing the end-user dashboard.

## File Structure
- `models/`: Stores the pre-trained `.pkl` model files (e.g., `no_show_rf.pkl`, `occupancy_rf.pkl`).
- `data/raw/`: Stores the generated CSV datasets.
- `notebooks/`: Contains the Jupyter notebook used for initial model development and EDA.
- `docs/`: Contains all analytical and technical documentation.

## Deployment Instructions
1. Clone the repository and initialize the virtual environment.
2. Install requirements using `pip install -r requirements.txt`.
3. Set your `GROQ_API_KEY` in the `.env` file.
4. **Important**: Run `python -m src.train_models` to generate the `.pkl` files into the `models/` folder.
5. Launch the application: `streamlit run app.py`.
