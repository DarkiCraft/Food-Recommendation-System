# SPEC-1-Personalized Food Recommendation System

## Background

Most food delivery platforms prioritize operational convenience (e.g., showing what is open, nearby, or popular) rather than understanding *user intent*. This results in repetitive, non-personalized landing pages that fail to adapt to individual preferences or situational context.

This project explores the design of a **personalized, context-aware food recommendation system** that learns from user interactions (orders, clicks, ratings) and adapts recommendations dynamically. The focus is not production scalability, but demonstrating sound recommender-system architecture, clean separation of concerns, and thoughtful use of machine learning concepts.

The system is designed as a prototype suitable for academic evaluation, with a Streamlit-based UI and a modular backend that can evolve as complexity is introduced later.## Requirements

### Must Have
- Track user interaction history (orders, clicks, ratings)
- Maintain user profiles and item (food/restaurant) metadata
- Support implicit and explicit feedback signals
- Generate personalized Top-N food recommendations per user
- Filter recommendations based on availability and basic context
- Provide a simple UI to display recommendations (Streamlit)### Should Have
- Hybrid recommendation approach (collaborative + content-based)
- Periodic model updates using historical interaction data
- Cold-start handling for new users and new items
- Logging of recommendation impressions and user responses

### Could Have
- Context-aware re-ranking (time of day, day of week)
- Multiple recommendation strategies (e.g., personalized vs trending)
- Basic visualization of learned user/item embeddings

### Won’t Have (for this phase)
- Real-time model retraining
- Large-scale optimization or distributed systems
- Advanced deep learning models

*Assumption:* The system operates on a small-to-medium synthetic dataset suitable for local experimentation and demonstration.## Method

### Overall Approach

The system follows a **hybrid recommendation architecture** that combines collaborative filtering with content-based and contextual signals. The design prioritizes **clarity, modularity, and ease of implementation**, while still demonstrating modern recommender-system concepts.

Rather than a single complex model, the system is composed of simple, well-scoped components that work together:
- Interaction logging and feature extraction
- Offline model training
- Online recommendation generation with filtering and ranking

Advanced models are only introduced if they **reduce implementation effort** or **improve conceptual clarity**.### High-Level Architecture

1. **User Interaction Layer (UI – Streamlit)**
   - User browses restaurants and food items
   - User places orders, clicks items, or provides ratings
   - Landing page requests personalized recommendations

2. **Event Logging & Storage**
   - All interactions are logged as events:
     - user_id, item_id, interaction_type, timestamp
   - Contextual attributes (time of day, day of week) are derived from timestamps
   - Data is stored in simple relational tables or CSV/Parquet files

3. **Feature & Dataset Builder (Offline)**
   - Aggregates interaction history into a user–item interaction matrix
   - Converts implicit feedback (orders, clicks) into weighted signals
   - Prepares optional item features (cuisine, price range)

4. **Recommendation Model (Offline Training)**
   - **Primary model:** Matrix Factorization for implicit feedback
     - Learns low-dimensional user and item embeddings
     - Simple to implement using existing Python libraries
   - **Fallback model:** Content-based similarity
     - Used for cold-start users or items

5. **Recommendation Service (Online / On-Demand)**
   - Loads trained embeddings into memory
   - For a given user:
     - Generates candidate items from the model
     - Filters items by availability and basic context
     - Ranks candidates by predicted preference score

6. **Feedback Loop**
   - User interactions with recommended items are logged
   - New data is included in the next training cycle
   - Model is retrained periodically (e.g., daily or manually)

### Recommendation Strategy

The system supports multiple strategies that can be switched or compared:
- Personalized (collaborative filtering)
- Content-based (similar cuisine or price)
- Popular / trending (baseline)

This allows qualitative comparison of recommendation quality and demonstrates extensibility without additional complexity.

### Why This Design

- **Simple models, strong architecture:** avoids overengineering
- **Explainable:** embeddings and similarities can be visualized
- **Extensible:** easy to add context-aware re-ranking later
- **Prototype-friendly:** suitable for a semester project and Streamlit demo

## Model Choice & Justification

### Primary Model: Neural Collaborative Filtering (NCF)

The core recommendation model is **Neural Collaborative Filtering (NCF)**, which extends traditional matrix factorization by learning user–item interactions through a shallow neural network.

In this system, users and items are represented as embedding vectors. These embeddings are combined (via concatenation or element-wise product) and passed through a small multi-layer perceptron to predict user preference. **Why NCF was chosen as the primary model:**
- Builds directly on matrix factorization concepts while being more expressive
- Easy to implement using modern deep learning libraries
- Demonstrates use of neural models without excessive complexity
- Widely recognized as a modern recommender-system approach

The model remains intentionally shallow to preserve interpretability and fast experimentation.### Supporting Model: Matrix Factorization

Traditional matrix factorization is retained as a **supporting and baseline model**:
- Used for comparison against NCF
- Serves as a simpler fallback if training data is sparse
- Helps illustrate the benefits of non-linear interaction modeling

This pairing highlights the architectural evolution from linear to neural methods.### Feedback Type: Implicit Feedback

The system primarily relies on **implicit feedback** such as:
- Orders (high weight)
- Clicks (medium weight)

Implicit feedback is preferred because:
- It is naturally generated by user behavior
- Explicit ratings are sparse and optional
- Interaction strength can be encoded as weighted signals

### Cold-Start Handling: Content-Based Similarity

NCF and matrix factorization both struggle with new users and new items. To address this, a **content-based similarity recommender** is used as a complementary strategy.

Item similarity is computed using metadata such as:
- Cuisine type
- Price range
- Restaurant attributes

This allows:
- Reasonable recommendations for first-time users
- Immediate exposure of new food items

### Baseline Model: Popularity-Based Ranking

A popularity-based recommender is included as a non-personalized baseline:
- Ranks items by global order or click frequency
- Provides a point of comparison for evaluation
- Acts as a safe fallback when user data is unavailable

## Implementation

This section outlines a concrete and lightweight implementation plan aligned with the project’s prototype nature. The goal is to keep the system **fully implementable**, while avoiding unnecessary production complexity.### Data Storage

For simplicity and transparency, data can be stored using:
- CSV or Parquet files during early experimentation
- SQLite or a lightweight relational database for structured access

#### Core Tables

**Users**
- user_id
- signup_date

**Items (Food / Restaurants)**
- item_id
- restaurant_id
- cuisine_type
- price_range
- availability_status

**Interactions**
- user_id
- item_id
- interaction_type (click, order)
- timestamp

**Ratings**
- user_id
- item_id
- rating (1–5)
- timestamp

Ratings are optional but, when present, are treated as **high-quality explicit feedback**.### Interaction Processing

Implicit interactions are converted into weighted signals:
- Order → high weight
- Click → medium weight

If a rating exists for a user–item pair:
- The rating is normalized (e.g., 1–5 → 0–1)
- It is incorporated into the training signal as an additional feature or increased interaction weight

This allows the model to benefit from explicit feedback **without depending on it**.

### Model Training Workflow

1. Aggregate interaction and rating data into a unified training dataset
2. Encode users and items as integer indices
3. Train a **shallow Neural Collaborative Filtering model**:
   - User embedding layer
   - Item embedding layer
   - Concatenation of embeddings
   - 2–3 dense layers
   - Sigmoid output for preference score
4. Train supporting matrix factorization model for comparison
5. Persist learned embeddings and model weights to disk

Training is performed offline and re-run periodically or manually.

### Recommendation Generation (Online)

When a user requests the landing page:
1. Load trained NCF model and embeddings
2. Generate candidate items
3. Filter candidates by:
   - Availability
   - Basic context (e.g., time of day)
4. Score each candidate using the NCF model
5. Re-rank using auxiliary signals:
   - Average item rating (to avoid surfacing poorly rated items)
6. Return Top-N items

### UI Integration (Streamlit)

The Streamlit UI provides:
- Personalized landing page recommendations
- Visible average ratings for items
- Post-order rating prompt (optional)
- Ability to switch between recommendation strategies (personalized vs popular)

Ratings are displayed to inform users and simultaneously logged to improve future recommendations.

### Feedback Loop

All user actions are continuously logged:
- Recommendation impressions
- Clicks
- Orders
- Ratings

These signals feed into subsequent training runs, closing the feedback loop and improving recommendation quality over time.
