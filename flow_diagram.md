# Pwani Oil Marketing Content Generator - System Flow Diagram

```mermaid
graph TB
    subgraph UI[User Interface]
        A[User Input] --> B[Streamlit Interface]
        B --> C[Campaign Details]
        B --> D[Target Market]
        B --> E[Advanced Settings]
    end

    subgraph RAG[RAG System]
        F[FAISS Vector Store] --> G[Hybrid Retriever]
        H[BM25 Retriever] --> G
        I[Context Cache] --> G
        G --> J[Context Compression]
    end

    subgraph Workflow[Workflow Engine]
        K[LangGraph Orchestrator] --> L[State Management]
        L --> M[Content Generation]
        M --> N[Error Handling]
    end

    subgraph Chat[Chat System]
        O[Chat History] --> P[Message Handler]
        P --> Q[Response Generator]
        Q --> R[Content Formatter]
    end

    subgraph Content[Content Pipeline]
        S[Template Processing] --> T[Context Enrichment]
        T --> U[LLM Generation]
        U --> V[Output Formatting]
    end

    B --> O
    C --> K
    G --> T
    M --> Q
    R --> V

    classDef primary fill:#f9f,stroke:#333,stroke-width:2px
    classDef secondary fill:#bbf,stroke:#333,stroke-width:2px
    classDef tertiary fill:#bfb,stroke:#333,stroke-width:2px

    class A,B,C,D,E primary
    class F,G,H,I,J secondary
    class K,L,M,N,O,P,Q,R tertiary
    class S,T,U,V secondary
```

## Component Descriptions

### User Interface
- **User Input**: Handles initial user requests and parameters
- **Streamlit Interface**: Main web interface for user interaction
- **Campaign Details**: Campaign configuration and parameters
- **Target Market**: Market segmentation and targeting options
- **Advanced Settings**: System configuration and LLM parameters

### RAG System
- **FAISS Vector Store**: Efficient vector storage and retrieval
- **Hybrid Retriever**: Combines BM25 and vector search
- **Context Cache**: Caches frequently accessed contexts
- **Context Compression**: Optimizes retrieved context

### Workflow Engine
- **LangGraph Orchestrator**: Manages content generation flow
- **State Management**: Handles workflow state and transitions
- **Content Generation**: Core content creation process
- **Error Handling**: Manages errors and fallbacks

### Chat System
- **Chat History**: Maintains conversation context
- **Message Handler**: Processes incoming/outgoing messages
- **Response Generator**: Generates contextual responses
- **Content Formatter**: Formats output for display

### Content Pipeline
- **Template Processing**: Handles content templates
- **Context Enrichment**: Enhances content with context
- **LLM Generation**: Language model interaction
- **Output Formatting**: Final content formatting