# System Design Document: Public Grievance Pattern Detector

## System Architecture Overview

The Public Grievance Pattern Detector follows a microservices-based architecture with a clear data pipeline from complaint ingestion to visualization. The system is designed for scalability, real-time processing, and efficient pattern detection.

### High-Level Architecture

```
┌─────────────────┐
│  Input Sources  │
│ (Web/App/SMS)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ingestion API   │
│   (FastAPI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Message Queue   │
│    (Redis)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NLP Processor   │
│   (Celery)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │
│   Generator     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Database │
│    (FAISS)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Clustering    │
│    Engine       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trend Detector  │
│ & Risk Scorer   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │
│   + PostGIS     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dashboard API  │
│   (FastAPI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frontend UI    │
│   (React.js)    │
└─────────────────┘
```

### Architecture Flow

**Complaint Input → Preprocessing → Embedding → Vector Storage → Clustering → Trend Detection → Risk Analysis → Dashboard**

## Component Design

### 1. Data Ingestion API

**Technology**: FastAPI

**Responsibilities**:
- Accept complaints from multiple channels (REST API, webhooks)
- Validate input data and sanitize text
- Extract metadata (timestamp, location, category)
- Assign unique complaint IDs
- Push to message queue for async processing

**Endpoints**:

```python
POST /api/v1/complaints
GET /api/v1/complaints/{id}
GET /api/v1/complaints?location={loc}&date_from={date}
POST /api/v1/complaints/batch
```

**Input Schema**:

```json
{
  "complaint_text": "string (required)",
  "location": {
    "address": "string",
    "latitude": "float",
    "longitude": "float"
  },
  "category": "string (optional)",
  "contact": {
    "name": "string",
    "phone": "string",
    "email": "string"
  },
  "source": "web|mobile|sms|email",
  "language": "string (auto-detected if not provided)"
}
```

**Processing Flow**:
1. Receive complaint via HTTP POST
2. Validate required fields
3. Geocode address if coordinates not provided
4. Store raw complaint in PostgreSQL
5. Publish to Redis queue for NLP processing
6. Return complaint ID to client

### 2. NLP Preprocessing

**Technology**: Celery workers with spaCy/NLTK

**Responsibilities**:
- Text cleaning and normalization
- Language detection and translation
- Tokenization and lemmatization
- Named Entity Recognition (NER)
- Sentiment analysis
- Keyword extraction

**Processing Pipeline**:

```python
def preprocess_complaint(complaint_text):
    # 1. Clean text
    text = remove_special_chars(complaint_text)
    text = normalize_whitespace(text)
    
    # 2. Language detection
    language = detect_language(text)
    if language != 'en':
        text = translate_to_english(text)
    
    # 3. Tokenization
    tokens = tokenize(text)
    
    # 4. Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # 5. Lemmatization
    tokens = lemmatize(tokens)
    
    # 6. NER - Extract entities
    entities = extract_entities(text)
    # entities: {locations, departments, dates, issues}
    
    # 7. Sentiment analysis
    sentiment = analyze_sentiment(text)
    
    # 8. Keyword extraction
    keywords = extract_keywords(text, top_n=10)
    
    return {
        'processed_text': ' '.join(tokens),
        'entities': entities,
        'sentiment': sentiment,
        'keywords': keywords,
        'language': language
    }
```

**Output**: Structured data stored in PostgreSQL with processed text ready for embedding

### 3. Embedding Generation

**Technology**: Sentence Transformers (SBERT) or OpenAI Embeddings

**Responsibilities**:
- Convert processed text to dense vector representations
- Generate semantic embeddings for similarity search
- Handle batch processing for efficiency

**Model Selection**:
- **Option 1**: `sentence-transformers/all-MiniLM-L6-v2` (lightweight, fast)
- **Option 2**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (multilingual)
- **Option 3**: OpenAI `text-embedding-ada-002` (high quality, API-based)

**Implementation**:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(processed_text):
    embedding = model.encode(processed_text)
    return embedding  # Returns 384-dimensional vector
```

**Optimization**:
- Batch processing (32-64 complaints at once)
- GPU acceleration if available
- Caching for repeated queries

### 4. Vector Database

**Technology**: FAISS (Facebook AI Similarity Search)

**Responsibilities**:
- Store high-dimensional embeddings
- Perform fast similarity search
- Support approximate nearest neighbor (ANN) queries
- Enable real-time clustering

**Architecture**:

```python
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension=384):
        self.dimension = dimension
        # Use IndexFlatL2 for exact search (small scale)
        # Use IndexIVFFlat for approximate search (large scale)
        self.index = faiss.IndexFlatL2(dimension)
        self.complaint_ids = []
    
    def add_embeddings(self, embeddings, complaint_ids):
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        self.complaint_ids.extend(complaint_ids)
    
    def search_similar(self, query_embedding, k=10):
        query = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query, k)
        similar_ids = [self.complaint_ids[i] for i in indices[0]]
        return similar_ids, distances[0]
```

**Persistence**:
- Save FAISS index to disk periodically
- Rebuild index daily with updated embeddings
- Maintain mapping between FAISS indices and complaint IDs

### 5. Clustering Engine

**Technology**: scikit-learn (DBSCAN, HDBSCAN, K-Means)

**Responsibilities**:
- Group similar complaints into clusters
- Identify cluster themes automatically
- Update clusters incrementally
- Detect outliers

**Algorithm Selection**:

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| DBSCAN | Density-based, arbitrary shapes | No need to specify k, finds outliers | Sensitive to parameters |
| HDBSCAN | Hierarchical density-based | Better than DBSCAN, automatic parameter selection | Slower |
| K-Means | Known number of categories | Fast, simple | Requires k, assumes spherical clusters |

**Implementation**:

```python
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

def cluster_complaints(embeddings, min_cluster_size=5):
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(embeddings_scaled)
    
    # -1 indicates outliers/noise
    return cluster_labels, clusterer.probabilities_
```

**Cluster Labeling**:

```python
def generate_cluster_label(complaint_texts):
    # Extract most common keywords
    keywords = extract_top_keywords(complaint_texts, top_n=5)
    
    # Use LLM for better labeling (optional)
    prompt = f"Generate a short label for complaints about: {keywords}"
    label = llm_generate(prompt)
    
    return label
```

**Scheduling**:
- Run clustering every 6 hours
- Incremental updates for new complaints
- Full re-clustering daily

### 6. Trend Detection

**Technology**: Statistical analysis with pandas and scipy

**Responsibilities**:
- Monitor complaint frequency over time
- Detect anomalies and spikes
- Identify seasonal patterns
- Compare against historical baselines

**Methods**:

**A. Time Series Analysis**:

```python
import pandas as pd
from scipy import stats

def detect_spike(complaint_counts, window=7):
    # Calculate rolling mean and std
    rolling_mean = complaint_counts.rolling(window=window).mean()
    rolling_std = complaint_counts.rolling(window=window).std()
    
    # Z-score for anomaly detection
    z_scores = (complaint_counts - rolling_mean) / rolling_std
    
    # Flag spikes (z-score > 2)
    spikes = z_scores > 2
    
    return spikes
```

**B. Frequency Analysis**:

```python
def analyze_trends(complaints_df):
    # Group by date and category
    daily_counts = complaints_df.groupby(
        [pd.Grouper(key='timestamp', freq='D'), 'category']
    ).size().reset_index(name='count')
    
    # Calculate percentage change
    daily_counts['pct_change'] = daily_counts.groupby('category')['count'].pct_change()
    
    # Detect significant increases (>50% increase)
    alerts = daily_counts[daily_counts['pct_change'] > 0.5]
    
    return alerts
```

**C. Geographic Hotspot Detection**:

```python
from scipy.spatial import distance_matrix

def detect_geographic_clusters(locations, threshold_km=2):
    # Calculate distance matrix
    dist_matrix = distance_matrix(locations, locations)
    
    # Find dense areas
    density = (dist_matrix < threshold_km).sum(axis=1)
    
    # Flag high-density locations
    hotspots = locations[density > density.mean() + 2*density.std()]
    
    return hotspots
```

### 7. Risk Scoring

**Technology**: Custom scoring algorithm with configurable weights

**Responsibilities**:
- Calculate risk scores for complaints and locations
- Prioritize high-risk issues
- Generate alerts for critical situations

**Risk Score Formula**:

```python
def calculate_risk_score(complaint_data):
    # Factors (0-10 scale each)
    frequency_score = min(complaint_data['count'] / 10, 10)
    recency_score = 10 * np.exp(-complaint_data['days_since_last'] / 7)
    severity_score = complaint_data['severity']  # From sentiment/keywords
    cluster_size_score = min(complaint_data['cluster_size'] / 20, 10)
    geographic_density = min(complaint_data['nearby_complaints'] / 15, 10)
    
    # Weighted combination
    weights = {
        'frequency': 0.25,
        'recency': 0.20,
        'severity': 0.25,
        'cluster_size': 0.15,
        'geographic_density': 0.15
    }
    
    risk_score = (
        weights['frequency'] * frequency_score +
        weights['recency'] * recency_score +
        weights['severity'] * severity_score +
        weights['cluster_size'] * cluster_size_score +
        weights['geographic_density'] * geographic_density
    )
    
    return risk_score
```

**Risk Categories**:
- **Critical (8-10)**: Immediate action required
- **High (6-8)**: Urgent attention needed
- **Medium (4-6)**: Monitor closely
- **Low (0-4)**: Routine handling

**Alert Generation**:

```python
def generate_alerts(risk_scores):
    alerts = []
    
    for location, score in risk_scores.items():
        if score >= 8:
            alerts.append({
                'level': 'CRITICAL',
                'location': location,
                'score': score,
                'message': f'Critical situation detected in {location}',
                'recommended_action': 'Immediate inspection required'
            })
    
    return alerts
```

## Data Flow Example

### End-to-End Processing of a Single Complaint

**Step 1: Complaint Submission**
```
User submits: "Water supply is very irregular in Sector 12. 
We get water only 2 hours a day. Very dirty water."
Location: Sector 12, Latitude: 28.5355, Longitude: 77.3910
```

**Step 2: Ingestion API**
```json
{
  "complaint_id": "CMP-2026-02-15-001234",
  "timestamp": "2026-02-15T08:30:00Z",
  "status": "received"
}
```

**Step 3: NLP Preprocessing**
```python
{
  "processed_text": "water supply irregular sector get water hour day dirty water",
  "entities": {
    "location": ["Sector 12"],
    "issue_type": ["water supply", "irregular", "dirty water"],
    "duration": ["2 hours a day"]
  },
  "sentiment": -0.65,  # Negative
  "keywords": ["water", "supply", "irregular", "dirty", "sector"],
  "language": "en"
}
```

**Step 4: Embedding Generation**
```python
embedding = [0.023, -0.145, 0.678, ..., 0.234]  # 384-dimensional vector
```

**Step 5: Vector Storage & Similarity Search**
```python
similar_complaints = [
  "CMP-2026-02-14-001180",  # distance: 0.12
  "CMP-2026-02-13-001095",  # distance: 0.18
  "CMP-2026-02-15-001201"   # distance: 0.21
]
# All related to water issues in Sector 12
```

**Step 6: Clustering**
```python
cluster_assignment = {
  "cluster_id": "CLU-WATER-SECTOR12-001",
  "cluster_label": "Water Supply Issues - Sector 12",
  "cluster_size": 47,
  "confidence": 0.89
}
```

**Step 7: Trend Detection**
```python
trend_analysis = {
  "spike_detected": True,
  "current_count": 47,
  "baseline_count": 12,
  "percentage_increase": 291.67,
  "time_window": "14 days"
}
```

**Step 8: Risk Scoring**
```python
risk_assessment = {
  "risk_score": 8.5,
  "risk_level": "CRITICAL",
  "factors": {
    "frequency": 9.4,
    "recency": 9.8,
    "severity": 7.5,
    "cluster_size": 8.2,
    "geographic_density": 8.9
  }
}
```

**Step 9: Dashboard Update**
```
Alert generated: "HIGH PRIORITY: Sector 12 experiencing systemic water supply issues"
Map updated: Red marker on Sector 12
Notification sent to Water Department
```

## Database Design

### PostgreSQL Schema

**Table: complaints**
```sql
CREATE TABLE complaints (
    complaint_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    raw_text TEXT NOT NULL,
    processed_text TEXT,
    location_address VARCHAR(255),
    location_point GEOGRAPHY(POINT, 4326),
    category VARCHAR(100),
    source VARCHAR(50),
    language VARCHAR(10),
    sentiment_score FLOAT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_complaints_timestamp ON complaints(timestamp);
CREATE INDEX idx_complaints_category ON complaints(category);
CREATE INDEX idx_complaints_location ON complaints USING GIST(location_point);
```

**Table: complaint_entities**
```sql
CREATE TABLE complaint_entities (
    id SERIAL PRIMARY KEY,
    complaint_id VARCHAR(50) REFERENCES complaints(complaint_id),
    entity_type VARCHAR(50),  -- location, department, issue_type
    entity_value VARCHAR(255),
    confidence FLOAT
);

CREATE INDEX idx_entities_complaint ON complaint_entities(complaint_id);
CREATE INDEX idx_entities_type ON complaint_entities(entity_type);
```

**Table: embeddings**
```sql
CREATE TABLE embeddings (
    complaint_id VARCHAR(50) PRIMARY KEY REFERENCES complaints(complaint_id),
    embedding_vector FLOAT[],  -- Store as array or use pgvector extension
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Table: clusters**
```sql
CREATE TABLE clusters (
    cluster_id VARCHAR(50) PRIMARY KEY,
    cluster_label VARCHAR(255),
    description TEXT,
    size INT,
    centroid FLOAT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Table: cluster_assignments**
```sql
CREATE TABLE cluster_assignments (
    id SERIAL PRIMARY KEY,
    complaint_id VARCHAR(50) REFERENCES complaints(complaint_id),
    cluster_id VARCHAR(50) REFERENCES clusters(cluster_id),
    confidence FLOAT,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cluster_assignments_complaint ON cluster_assignments(complaint_id);
CREATE INDEX idx_cluster_assignments_cluster ON cluster_assignments(cluster_id);
```

**Table: risk_scores**
```sql
CREATE TABLE risk_scores (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50),  -- complaint, location, cluster
    entity_id VARCHAR(255),
    risk_score FLOAT,
    risk_level VARCHAR(20),
    factors JSONB,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_scores_entity ON risk_scores(entity_type, entity_id);
CREATE INDEX idx_risk_scores_level ON risk_scores(risk_level);
```

**Table: alerts**
```sql
CREATE TABLE alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    title VARCHAR(255),
    description TEXT,
    location VARCHAR(255),
    related_cluster_id VARCHAR(50),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_severity ON alerts(severity);
```

**Table: trend_metrics**
```sql
CREATE TABLE trend_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE,
    category VARCHAR(100),
    location VARCHAR(255),
    complaint_count INT,
    percentage_change FLOAT,
    is_spike BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trend_metrics_date ON trend_metrics(metric_date);
CREATE INDEX idx_trend_metrics_category ON trend_metrics(category);
```

## Scalability Considerations

### Horizontal Scaling

**API Layer**:
- Deploy multiple FastAPI instances behind load balancer (Nginx/HAProxy)
- Stateless design for easy scaling
- Use Redis for session management if needed

**Processing Layer**:
- Scale Celery workers independently based on queue depth
- Separate worker pools for different task types (NLP, embedding, clustering)
- Auto-scaling based on CPU/memory metrics

**Database Layer**:
- PostgreSQL read replicas for dashboard queries
- Connection pooling (PgBouncer)
- Partition large tables by date
- Archive old complaints to cold storage

### Caching Strategy

**Redis Caching**:
```python
# Cache frequently accessed data
- Recent complaints (TTL: 5 minutes)
- Cluster summaries (TTL: 1 hour)
- Risk scores (TTL: 30 minutes)
- Dashboard aggregations (TTL: 15 minutes)
```

**FAISS Index Caching**:
- Keep index in memory for fast queries
- Periodic snapshots to disk
- Lazy loading for large indices

### Performance Optimization

**Batch Processing**:
- Process embeddings in batches of 32-64
- Bulk insert to database
- Batch similarity searches

**Async Processing**:
- Use async/await in FastAPI
- Non-blocking I/O operations
- Background tasks for non-critical operations

**Query Optimization**:
- Materialized views for dashboard metrics
- Pre-computed aggregations
- Efficient indexing strategy

### Data Retention

**Hot Data** (0-3 months):
- Full text search enabled
- Real-time clustering
- Stored in primary PostgreSQL

**Warm Data** (3-12 months):
- Compressed storage
- Batch processing only
- Moved to separate partition

**Cold Data** (>12 months):
- Archive to S3/object storage
- Embeddings removed from FAISS
- Available for historical analysis only

## Future Enhancements

### Phase 2 Features

**1. Predictive Analytics**
- Forecast complaint volumes using time series models (ARIMA, Prophet)
- Predict emerging issues before they spike
- Resource allocation optimization

**2. Multi-Modal Analysis**
- Image analysis for complaints with photos
- Audio transcription for voice complaints
- Video analysis for evidence

**3. Advanced NLP**
- Fine-tuned domain-specific language models
- Multi-lingual support without translation
- Emotion detection beyond sentiment

**4. Automated Response**
- Chatbot for complaint status updates
- Auto-categorization and routing
- Suggested resolutions based on similar cases

**5. Integration Ecosystem**
- Mobile app for citizens
- SMS gateway for feature phones
- Social media monitoring (Twitter, Facebook)
- Integration with existing municipal systems

### Phase 3 Features

**1. Root Cause Analysis**
- Causal inference models
- Infrastructure correlation analysis
- Policy impact assessment

**2. Citizen Engagement**
- Feedback loop on resolutions
- Community voting on priorities
- Transparency dashboard for public

**3. Advanced Visualization**
- 3D heatmaps
- Temporal animation of complaint spread
- Network graphs showing issue relationships

**4. AI-Powered Insights**
- LLM-generated executive summaries
- Natural language query interface
- Automated report generation

**5. Cross-Department Collaboration**
- Shared dashboard across departments
- Workflow automation
- SLA tracking and enforcement

### Technical Debt & Improvements

**Short Term**:
- Comprehensive unit and integration tests
- API rate limiting and throttling
- Enhanced error handling and logging
- Security audit and penetration testing

**Medium Term**:
- Migrate to microservices architecture
- Implement event sourcing for audit trail
- Add GraphQL API for flexible queries
- Real-time streaming with Apache Kafka

**Long Term**:
- Multi-tenancy for different cities
- Federated learning across municipalities
- Blockchain for complaint immutability
- Edge computing for offline capability

## Deployment Architecture

### Development Environment
```
Docker Compose:
- FastAPI (port 8000)
- PostgreSQL (port 5432)
- Redis (port 6379)
- Celery workers
- React dev server (port 3000)
```

### Production Environment
```
Kubernetes Cluster:
- API pods (3 replicas)
- Worker pods (5 replicas, auto-scaling)
- PostgreSQL (managed service)
- Redis (managed service)
- FAISS service (dedicated pods)
- Frontend (CDN + static hosting)
- Load balancer (Ingress)
- Monitoring (Prometheus + Grafana)
```

### CI/CD Pipeline
```
1. Code push to GitHub
2. Run tests (pytest, Jest)
3. Build Docker images
4. Push to container registry
5. Deploy to staging
6. Run integration tests
7. Manual approval
8. Deploy to production
9. Health checks
10. Rollback on failure
```

---

**Document Version**: 1.0  
**Last Updated**: February 15, 2026  
**Status**: Ready for Implementation
