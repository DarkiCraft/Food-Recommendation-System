# Personalized Food Recommendation System

A prototype recommender system designed to demonstrate a hybrid recommendation architecture, combining Neural Collaborative Filtering (NCF), Matrix Factorization, and Content-based approaches.

## Features

- **Personalized Recommendations**: "For You" (NCF/MF), "Trending" (Popularity), and "Based on Taste" (Content-based).
- **Hybrid Architecture**: Leverages both collaborative filtering (user interactions) and content-based signals (cold-start handling).
- **Interaction Tracking**: Logs user clicks, orders, and ratings to refine the model.
- **Admin Dashboard**: Visualize system metrics, matrix sparsity, and simulate user behavior.
- **Interactive UI**: Built with Streamlit for a responsive and demonstrative user experience.

## Getting Started

### Prerequisites

- Python 3.8+
- [Pip](https://pip.pypa.io/en/stable/)

### Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the Streamlit application:

```bash
streamlit run src/ui/app.py
```

- **User Login**: Interact with the system as a user to generate data.
- **Admin Access**: Log in with user ID `admin` to access the dashboard and simulation tools.

## Architecture

- **Frontend**: Streamlit
- **Backend**: Python, PyTorch (Neural Networks), Scikit-Learn (Matrix Factorization)
- **Data**: Synthetic dataset generation for prototyping.

## License

See [LICENSE](LICENSE) file.
