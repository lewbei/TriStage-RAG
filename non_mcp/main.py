#!/usr/bin/env python3
"""
3-Stage Document Retrieval System - Standalone Non-MCP Application

This is a complete standalone application that implements the 3-stage retrieval pipeline:
Stage 1: Fast candidate generation using embeddings and BM25
Stage 2: Multi-vector rescoring for improved relevance  
Stage 3: Cross-encoder reranking for final ranking

Features:
- Command-line interface for document management and search
- Uses local models from ./models/ directory
- Real-time search with detailed results
- Document persistence and management
- Performance monitoring

Usage:
    python main.py
"""

import sys
import os
import json
import time
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Allow running directly without `pip install -e .`; no-op once installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import pipeline components
from tristage_rag.stage1_retriever import Stage1Retriever, Stage1Config
from tristage_rag.stage2_rescorer import ColBERTScorer, Stage2Config
from tristage_rag.stage3_reranker import CrossEncoderReranker, Stage3Config


@dataclass
class AppConfig:
    """Application configuration"""
    models_dir: str = "../models"
    data_dir: str = "../data"
    index_dir: str = "../faiss_index"
    max_results: int = 20
    enable_bm25: bool = True
    device: str = "auto"
    log_level: str = "INFO"


class DocumentManager:
    """Manages document storage and retrieval"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.documents_file = self.data_dir / "documents.pkl"
        self.metadata_file = self.data_dir / "metadata.json"
        self.documents = []
        self.metadata = {}
        self._load_documents()
    
    def _load_documents(self):
        """Load documents from disk"""
        if self.documents_file.exists():
            try:
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
            except Exception as e:
                logging.warning(f"Could not load documents: {e}")
                self.documents = []
        
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load metadata: {e}")
                self.metadata = {}
    
    def _save_documents(self):
        """Save documents to disk"""
        try:
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save documents: {e}")
    
    def add_documents(self, documents: List[str], source: str = "manual"):
        """Add documents to the collection"""
        new_count = 0
        for doc in documents:
            if doc and doc.strip() and doc not in self.documents:
                self.documents.append(doc.strip())
                new_count += 1
        
        if new_count > 0:
            self.metadata[f"last_update_{source}"] = time.time()
            self.metadata[f"count_{source}"] = self.metadata.get(f"count_{source}", 0) + new_count
            self.metadata["total_documents"] = len(self.documents)
            self._save_documents()
            logging.info(f"Added {new_count} documents from {source}")
        
        return new_count
    
    def get_documents(self) -> List[str]:
        """Get all documents"""
        return self.documents.copy()
    
    def clear_documents(self):
        """Clear all documents"""
        self.documents.clear()
        self.metadata.clear()
        self.metadata["total_documents"] = 0
        self._save_documents()
        logging.info("Cleared all documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document statistics"""
        return {
            "total_documents": len(self.documents),
            "metadata": self.metadata.copy(),
            "data_dir": str(self.data_dir)
        }


class ThreeStageRetrievalSystem:
    """Complete 3-stage retrieval system"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.doc_manager = DocumentManager(config.data_dir)
        self.stage1 = None
        self.stage2 = None
        self.stage3 = None
        
        # Performance tracking
        self.search_history = []
        
        # Initialize pipeline stages
        self._initialize_stages()

        # Load existing index if present; otherwise do nothing until user adds docs
        try:
            from pathlib import Path as _Path
            idx_dir = _Path(self.config.index_dir)
            idx_file = idx_dir / "stage1_index.pkl"
            if idx_file.exists():
                self.stage1.load_index(str(idx_file))
                self.logger.info("Loaded existing Stage 1 index from disk")
            # No auto-indexing; user controls ingestion via CLI or web UI
        except Exception as e:
            self.logger.warning(f"Index load/init skipped due to error: {e}")
    
    def _initialize_stages(self):
        """Initialize the three pipeline stages"""
        try:
            # Stage 1: Fast candidate generation
            stage1_config = Stage1Config(
                model_name="google/embeddinggemma-300m",
                device=self.config.device,
                cache_dir=self.config.models_dir,
                index_dir=self.config.index_dir,
                top_k_candidates=100,
                batch_size=16,
                enable_bm25=self.config.enable_bm25,
                use_fp16=True
            )
            self.stage1 = Stage1Retriever(stage1_config)
            self.logger.info("Stage 1 initialized successfully")

            # Stage 2: Multi-vector rescoring
            stage2_config = Stage2Config(
                model_name="lightonai/GTE-ModernColBERT-v1",
                device=self.config.device,
                cache_dir=self.config.models_dir,
                top_k_candidates=50,
                batch_size=8,
                max_seq_length=192,
                use_fp16=True
            )
            self.stage2 = ColBERTScorer(stage2_config)
            self.logger.info("Stage 2 initialized successfully")
            
            # Stage 3: Cross-encoder reranking
            stage3_config = Stage3Config(
                model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
                device=self.config.device,
                cache_dir=self.config.models_dir,
                top_k_final=self.config.max_results,
                batch_size=16,
                max_length=256,
                use_fp16=True
            )
            self.stage3 = CrossEncoderReranker(stage3_config)
            self.logger.info("Stage 3 initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline stages: {e}")
            raise
    
    def _index_documents(self, new_documents: Optional[List[str]] = None):
        """Index documents in Stage 1.

        If new_documents is provided, only those are added; otherwise all persisted docs are added.
        Always saves the index to disk after indexing.
        """
        try:
            from pathlib import Path as _Path
            idx_path = str(_Path(self.config.index_dir) / "stage1_index.pkl")
            if new_documents is not None:
                docs = [d for d in new_documents if d and d.strip()]
            else:
                docs = self.doc_manager.get_documents()
            if docs:
                self.stage1.add_documents(docs)
                self.stage1.save_index(idx_path)
                self.logger.info(f"Indexed {len(docs)} documents in Stage 1 and saved index")
        except Exception as e:
            self.logger.error(f"Error indexing documents: {e}")
    
    def add_documents(self, documents: List[str], source: str = "manual") -> int:
        """Add documents to the system"""
        # Track existing docs to compute newly added
        before_set = set(self.doc_manager.get_documents())
        new_count = self.doc_manager.add_documents(documents, source)

        if new_count > 0:
            # Index only new docs
            after_docs = self.doc_manager.get_documents()
            new_docs = [d for d in after_docs if d not in before_set]
            self._index_documents(new_docs)

        return new_count
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Perform 3-stage search"""
        if top_k is None:
            top_k = self.config.max_results
        
        start_time = time.time()
        
        try:
            # Stage 1: Fast candidate generation
            stage1_start = time.time()
            candidates = self.stage1.search(query, top_k=100)
            stage1_time = time.time() - stage1_start
            
            if not candidates:
                return {
                    "query": query,
                    "results": [],
                    "stage1_time": stage1_time,
                    "stage2_time": 0,
                    "stage3_time": 0,
                    "total_time": time.time() - start_time,
                    "candidate_count": 0,
                    "final_count": 0
                }
            
            # Stage 2: Multi-vector rescoring
            stage2_start = time.time()
            rescored = self.stage2.rescore_candidates(query, candidates[:50])  # Top 50 candidates
            stage2_time = time.time() - stage2_start
            
            # Stage 3: Cross-encoder reranking
            stage3_start = time.time()
            final_results = self.stage3.rerank(query, rescored[:20])  # Top 20 rescored
            stage3_time = time.time() - stage3_start
            
            # Prepare results (prefer Stage 3 score; fall back sensibly)
            results = []
            for i, result in enumerate(final_results[:top_k]):
                s1 = result.get("stage1_score")
                if s1 is None:
                    s1 = result.get("score", 0)
                s2 = result.get("stage2_score", 0)
                s3 = result.get("stage3_score", 0)
                final_s = s3 if s3 is not None else (s2 if s2 is not None else (s1 if s1 is not None else 0))
                results.append({
                    "rank": i + 1,
                    "doc_id": result.get("doc_id", f"doc_{i}"),
                    "document": result.get("document", ""),
                    "final_score": final_s,
                    "stage1_score": s1 if s1 is not None else 0,
                    "stage2_score": s2 if s2 is not None else 0,
                    "stage3_score": s3 if s3 is not None else 0
                })
            
            total_time = time.time() - start_time
            
            # Record search history
            search_record = {
                "query": query,
                "timestamp": time.time(),
                "total_time": total_time,
                "result_count": len(results),
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "stage3_time": stage3_time
            }
            self.search_history.append(search_record)
            
            # Keep only last 100 searches
            if len(self.search_history) > 100:
                self.search_history = self.search_history[-100:]
            
            return {
                "query": query,
                "results": results,
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "stage3_time": stage3_time,
                "total_time": total_time,
                "candidate_count": len(candidates),
                "final_count": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return {
                "query": query,
                "results": [],
                "stage1_time": 0,
                "stage2_time": 0,
                "stage3_time": 0,
                "total_time": time.time() - start_time,
                "candidate_count": 0,
                "final_count": 0,
                "error": str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        doc_stats = self.doc_manager.get_stats()
        
        return {
            "config": asdict(self.config),
            "document_stats": doc_stats,
            "search_count": len(self.search_history),
            "stages": {
                "stage1": {
                    "model": "google/embeddinggemma-300m",
                    "indexed": self.stage1 is not None and len(self.stage1.documents) > 0
                },
                "stage2": {
                    "model": "lightonai/GTE-ModernColBERT-v1", 
                    "ready": self.stage2 is not None
                },
                "stage3": {
                    "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                    "ready": self.stage3 is not None
                }
            }
        }
    
    def clear_all_data(self):
        """Clear all documents and reset system"""
        self.doc_manager.clear_documents()
        self.search_history.clear()
        # Remove persisted index files
        try:
            from pathlib import Path as _Path
            idx_dir = _Path(self.config.index_dir)
            (idx_dir / "stage1_index.pkl").unlink(missing_ok=True)
            (idx_dir / "stage1_faiss.index").unlink(missing_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to remove index files: {e}")

        # Reinitialize stages
        self._initialize_stages()
        
        self.logger.info("System cleared and reinitialized")


class CommandLineInterface:
    """Command-line interface for the retrieval system"""
    
    def __init__(self, system: ThreeStageRetrievalSystem):
        self.system = system
        self.running = True
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("3-Stage Document Retrieval System")
        print("="*60)
        print(f"Documents: {len(self.system.doc_manager.get_documents())}")
        print(f"Searches:  {len(self.system.search_history)}")
        print()
        print("1. Add documents manually")
        print("2. Load documents from file")
        print("3. Load documents from directory")
        print("4. View documents")
        print("5. Search documents")
        print("6. Batch search")
        print("7. System information")
        print("8. Export search history")
        print("9. Clear all data")
        print("0. Exit")
        print()
    
    def add_documents_manually(self):
        """Add documents manually"""
        print("\nAdd Documents Manually")
        print("Enter documents one by one. Type 'DONE' to finish.")
        
        documents = []
        while True:
            doc = input(f"Document {len(documents)+1}: ").strip()
            if doc.upper() == 'DONE':
                break
            if doc:
                documents.append(doc)
        
        if documents:
            count = self.system.add_documents(documents)
            print(f"Added {count} new documents.")
        else:
            print("No documents added.")
    
    def load_documents_from_file(self):
        """Load documents from file"""
        filepath = input("\nEnter file path: ").strip()
        
        if not Path(filepath).exists():
            print("File not found.")
            return
        
        try:
            documents = []
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        documents = [str(d) for d in data if d]
                    elif isinstance(data, dict) and 'documents' in data:
                        documents = data['documents']
                else:
                    # Text file - split by paragraphs
                    content = f.read()
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    documents = paragraphs
            
            count = self.system.add_documents(documents, source=filepath)
            print(f"Loaded {count} documents from {filepath}")
            
        except Exception as e:
            print(f"Error loading file: {e}")
    
    def load_documents_from_directory(self):
        """Load documents from directory"""
        dirpath = input("\nEnter directory path: ").strip()
        
        if not Path(dirpath).exists():
            print("Directory not found.")
            return
        
        try:
            documents = []
            dir_path = Path(dirpath)
            
            # Load from text files
            for txt_file in dir_path.rglob("*.txt"):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        documents.append(content)
            
            # Load from JSON files
            for json_file in dir_path.rglob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            documents.extend([str(d) for d in data if d])
                        elif isinstance(data, dict) and 'text' in data:
                            documents.append(data['text'])
                    except Exception as e:
                        logging.warning(f"Skipping malformed JSON file: {e}")
            
            count = self.system.add_documents(documents, source=dirpath)
            print(f"Loaded {count} documents from {dirpath}")
            
        except Exception as e:
            print(f"Error loading directory: {e}")
    
    def view_documents(self):
        """View current documents"""
        docs = self.system.doc_manager.get_documents()
        
        if not docs:
            print("\nNo documents loaded.")
            return
        
        print(f"\nDocuments ({len(docs)}):")
        print("-" * 50)
        
        for i, doc in enumerate(docs[:10], 1):
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"{i:2d}. {preview}")
        
        if len(docs) > 10:
            print(f"... and {len(docs) - 10} more documents")
    
    def search_documents(self):
        """Search documents"""
        query = input("\nEnter search query: ").strip()
        
        if not query:
            print("Query cannot be empty.")
            return
        
        try:
            top_k = int(input("Number of results (default 5): ") or "5")
        except ValueError:
            top_k = 5
        
        print(f"\nSearching for: '{query}'")
        results = self.system.search(query, top_k)
        
        self.display_search_results(results)
    
    def display_search_results(self, results: Dict[str, Any]):
        """Display search results"""
        print(f"\nSearch Results for: '{results['query']}'")
        print(f"Time: {results['total_time']:.3f}s "
              f"(S1: {results['stage1_time']:.3f}s, "
              f"S2: {results['stage2_time']:.3f}s, "
              f"S3: {results['stage3_time']:.3f}s)")
        print(f"Candidates: {results['candidate_count']}, Final: {results['final_count']}")
        print("-" * 70)
        
        if not results['results']:
            print("No results found.")
            return
        
        for result in results['results']:
            print(f"\nRank {result['rank']} (Score: {result['final_score']:.4f})")
            print(f"  S1: {result['stage1_score']:.4f} | "
                  f"S2: {result['stage2_score']:.4f} | "
                  f"S3: {result['stage3_score']:.4f}")
            print(f"  {result['document'][:200]}...")
    
    def batch_search(self):
        """Perform batch search"""
        print("\nBatch Search")
        print("Enter queries one by one. Type 'DONE' to finish.")
        
        queries = []
        while True:
            query = input(f"Query {len(queries)+1}: ").strip()
            if query.upper() == 'DONE':
                break
            if query:
                queries.append(query)
        
        if not queries:
            print("No queries provided.")
            return
        
        try:
            top_k = int(input("Results per query (default 3): ") or "3")
        except ValueError:
            top_k = 3
        
        print(f"\nProcessing {len(queries)} queries...")
        total_time = 0
        
        for query in queries:
            results = self.system.search(query, top_k)
            total_time += results['total_time']
            
            print(f"  '{query[:50]}...' -> {results['final_count']} results "
                  f"({results['total_time']:.3f}s)")
        
        print(f"\nBatch completed in {total_time:.3f}s "
              f"(avg: {total_time/len(queries):.3f}s per query)")
    
    def show_system_info(self):
        """Show system information"""
        info = self.system.get_system_info()
        
        print("\nSystem Information")
        print("=" * 50)
        
        print(f"Documents: {info['document_stats']['total_documents']}")
        print(f"Searches:  {info['search_count']}")
        print(f"Device:    {info['config']['device']}")
        print(f"Models Dir: {info['config']['models_dir']}")
        print(f"Data Dir:  {info['config']['data_dir']}")
        
        print("\nPipeline Stages:")
        print(f"  Stage 1: {info['stages']['stage1']['model']} "
              f"({'Indexed' if info['stages']['stage1']['indexed'] else 'Not indexed'})")
        print(f"  Stage 2: {info['stages']['stage2']['model']} "
              f"({'Ready' if info['stages']['stage2']['ready'] else 'Not ready'})")
        print(f"  Stage 3: {info['stages']['stage3']['model']} "
              f"({'Ready' if info['stages']['stage3']['ready'] else 'Not ready'})")
    
    def export_search_history(self):
        """Export search history"""
        if not self.system.search_history:
            print("\nNo search history to export.")
            return
        
        filename = input("\nEnter export filename (default: search_history.json): ").strip()
        if not filename:
            filename = "search_history.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        try:
            export_data = {
                "system_info": self.system.get_system_info(),
                "search_history": self.system.search_history,
                "export_timestamp": time.time()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"Exported {len(self.system.search_history)} searches to {filename}")
            
        except Exception as e:
            print(f"Export failed: {e}")
    
    def clear_all_data(self):
        """Clear all data"""
        confirm = input("\nAre you sure you want to clear all data? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            self.system.clear_all_data()
            print("All data cleared.")
        else:
            print("Operation cancelled.")
    
    def run(self):
        """Run the command-line interface"""
        print("3-Stage Document Retrieval System")
        print("Starting up...")
        
        while self.running:
            try:
                self.show_menu()
                choice = input("Enter your choice (0-9): ").strip()
                
                if choice == '1':
                    self.add_documents_manually()
                elif choice == '2':
                    self.load_documents_from_file()
                elif choice == '3':
                    self.load_documents_from_directory()
                elif choice == '4':
                    self.view_documents()
                elif choice == '5':
                    self.search_documents()
                elif choice == '6':
                    self.batch_search()
                elif choice == '7':
                    self.show_system_info()
                elif choice == '8':
                    self.export_search_history()
                elif choice == '9':
                    self.clear_all_data()
                elif choice == '0':
                    self.running = False
                    print("Goodbye!")
                else:
                    print("Invalid choice. Please enter 0-9.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                self.running = False
            except Exception as e:
                print(f"\nError: {e}")
                input("Press Enter to continue...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="3-Stage Document Retrieval System")
    # Default to repo-level directories when running from non_mcp folder
    parser.add_argument('--models-dir', default='../models', help='Models directory')
    parser.add_argument('--data-dir', default='../data', help='Data directory')
    parser.add_argument('--index-dir', default='../faiss_index', help='Index directory')
    parser.add_argument('--device', default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--webui', action='store_true', help='Launch the Flask Web UI')
    parser.add_argument('--webui-host', default='127.0.0.1', help='Web UI host (default 127.0.0.1)')
    parser.add_argument('--webui-port', type=int, default=5051, help='Web UI port (default 5051)')
    parser.add_argument('--query', help='Search query (command line mode)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--load', help='Load documents from file/dir')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--log-file', default='retrieval_pipeline.log', help='Log file path')
    parser.add_argument('--config', default=None, help='Optional YAML config to load settings')
    
    args = parser.parse_args()
    
    # Setup logging (file + console)
    try:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
    except Exception:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create configuration (optionally load from YAML)
    config = AppConfig(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        device=args.device,
        log_level=args.log_level.upper()
    )

    if args.config:
        try:
            import yaml
            from pathlib import Path as _Path
            cfg_path = _Path(args.config)
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                p = cfg.get('pipeline', {})
                # Override paths and logging if provided
                config.models_dir = p.get('cache_dir', config.models_dir)
                # Prefer repo-level faiss index naming if present
                config.index_dir = p.get('index_dir', config.index_dir)
                config.device = p.get('device', config.device)
                config.log_level = p.get('log_level', config.log_level)
                # If log file from config is provided and different, add a file handler
                log_file = p.get('log_file')
                if log_file:
                    # Add an extra file handler without duplicating console handler
                    root_logger = logging.getLogger()
                    root_logger.addHandler(logging.FileHandler(log_file))
                logging.info(f"Loaded settings from {cfg_path}")
        except Exception as e:
            logging.warning(f"Failed to load config {args.config}: {e}")
    
    # If asked to run the Web UI, delegate to the Flask app to avoid import cycles
    if args.webui:
        import os as _os, sys as _sys, subprocess as _sp
        # Pass device via env so webui/app.py picks it up
        env = _os.environ.copy()
        env['NON_MCP_DEVICE'] = config.device
        env['NON_MCP_WEBUI_LOG_LEVEL'] = config.log_level
        # Launch the web UI as a child process using the same interpreter
        webui_path = str(Path(__file__).parent / 'webui' / 'app.py')
        cmd = [_sys.executable, webui_path]
        try:
            print(f"Starting Web UI at http://{args.webui_host}:{args.webui_port} ...")
            # Allow host/port override via env for app.py
            env['NON_MCP_WEBUI_HOST'] = args.webui_host
            env['NON_MCP_WEBUI_PORT'] = str(args.webui_port)
            _sp.run(cmd, env=env, check=True)
        except _sp.CalledProcessError as e:
            print(f"Failed to start Web UI: {e}")
            sys.exit(e.returncode if hasattr(e, 'returncode') else 1)
        return

    # Initialize system
    try:
        system = ThreeStageRetrievalSystem(config)
        print("System initialized successfully!")
        
        # Load documents if specified
        if args.load:
            load_path = Path(args.load)
            if load_path.exists():
                if load_path.is_file():
                    with open(load_path, 'r', encoding='utf-8') as f:
                        if load_path.suffix == '.json':
                            data = json.load(f)
                            docs = data if isinstance(data, list) else data.get('documents', [])
                        else:
                            docs = [line.strip() for line in f if line.strip()]
                        count = system.add_documents(docs, source=str(load_path))
                        print(f"Loaded {count} documents from {load_path}")
                else:
                    # Directory loading
                    docs = []
                    for txt_file in load_path.rglob("*.txt"):
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                docs.append(content)
                    count = system.add_documents(docs, source=str(load_path))
                    print(f"Loaded {count} documents from {load_path}")
        
        # Single query mode
        if args.query:
            if not system.doc_manager.get_documents():
                print("No documents available for search.")
                return
            
            print(f"Searching for: '{args.query}'")
            results = system.search(args.query, args.top_k)
            
            # Display results
            cli = CommandLineInterface(system)
            cli.display_search_results(results)
            
        else:
            # Interactive mode
            cli = CommandLineInterface(system)
            cli.run()
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
