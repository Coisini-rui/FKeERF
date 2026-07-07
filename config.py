# ================================
# 深度学习蛋白质特征提取配置文件
# Deep Learning Protein Feature Extraction Configuration
# ================================

# 多尺度特征提取配置
MULTISCALE_FEATURE_CONFIG = {
    'feature_scales': {
        'atomic_scale': {
            'kernel_sizes': [1],
            'dilation_rates': [1],
            'feature_dim': 64,
            'use_attention': True
        },
        'short_range': {
            'kernel_sizes': [2, 3, 4],
            'dilation_rates': [1, 2],
            'feature_dim': 128,
            'attention_heads': 8
        },
        'medium_range': {
            'kernel_sizes': [5, 7, 9],
            'dilation_rates': [1, 3, 5],
            'feature_dim': 256,
            'attention_heads': 16
        },
        'long_range': {
            'kernel_sizes': [11, 15, 21],
            'dilation_rates': [1, 5, 10],
            'feature_dim': 512,
            'attention_heads': 32
        }
    },

    'physicochemical_properties': {
        'hydrophobicity': {
            'scale_type': 'kyte_doolittle',
            'normalization': 'minmax',
            'weight': 0.15
        },
        'polarity': {
            'scale_type': 'grantham',
            'normalization': 'zscore',
            'weight': 0.12
        },
        'volume': {
            'scale_type': 'zimmerman',
            'normalization': 'minmax',
            'weight': 0.10
        },
        'charge': {
            'scale_type': 'electrostatic',
            'normalization': 'standard',
            'weight': 0.18
        },
        'solvent_accessibility': {
            'scale_type': 'accessible_surface',
            'normalization': 'minmax',
            'weight': 0.15
        },
        'secondary_structure': {
            'scale_type': 'chou_fasman',
            'normalization': 'probability',
            'weight': 0.20
        },
        'flexibility': {
            'scale_type': 'karplus_schulz',
            'normalization': 'minmax',
            'weight': 0.10
        }
    },

    'feature_fusion': {
        'fusion_method': 'hierarchical_attention',
        'attention_layers': 3,
        'cross_scale_interaction': True,
        'residual_connections': True,
        'normalization': 'layer_norm',
        'dropout_rate': 0.1
    }
}

# 双通道神经网络架构配置
DUAL_CHANNEL_NETWORK_CONFIG = {
    'sequence_channel': {
        'encoder_type': 'transformer_xl',
        'hidden_dim': 1024,
        'num_layers': 12,
        'attention_heads': 32,
        'feedforward_dim': 4096,
        'activation': 'gelu',
        'dropout': 0.1,
        'use_preln': True,
        'max_sequence_length': 1024
    },

    'structure_channel': {
        'encoder_type': 'graph_transformer',
        'hidden_dim': 512,
        'num_layers': 8,
        'attention_heads': 16,
        'graph_neighbors': 30,
        'edge_feature_dim': 64,
        'geometric_attention': True,
        'use_3d_coordinates': True
    },

    'channel_fusion': {
        'fusion_strategy': 'adaptive_gating',
        'fusion_layers': 4,
        'cross_attention_heads': 24,
        'weight_initialization': {
            'sequence_weight': 0.6,
            'structure_weight': 0.4,
            'learnable_weights': True
        },
        'skip_connections': True,
        'fusion_dropout': 0.15
    }
}

# 多模态证据融合配置
MULTIMODAL_EVIDENCE_FUSION = {
    'fusion_methodology': {
        'primary_method': 'dempster_shafer',
        'fallback_method': 'weighted_average',
        'uncertainty_quantification': True,
        'confidence_calibration': True
    },

    'evidence_sources': {
        'sequence_evidence': {
            'weight': 0.35,
            'reliability_score': 0.85,
            'uncertainty_threshold': 0.25
        },
        'structure_evidence': {
            'weight': 0.30,
            'reliability_score': 0.78,
            'uncertainty_threshold': 0.30
        },
        'evolutionary_evidence': {
            'weight': 0.20,
            'reliability_score': 0.92,
            'uncertainty_threshold': 0.15
        },
        'physicochemical_evidence': {
            'weight': 0.15,
            'reliability_score': 0.75,
            'uncertainty_threshold': 0.35
        }
    },

    'decision_framework': {
        'decision_criterion': 'pignistic_probability',
        'confidence_threshold': 0.7,
        'conflict_resolution': 'yager_rule',
        'evidence_discounting': True,
        'dempster_combination_rounds': 3
    }
}

# 训练优化器配置
TRAINING_OPTIMIZATION_CONFIG = {
    'optimizer': {
        'type': 'lamb_optimizer',
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-6
    },

    'scheduler': {
        'type': 'cosine_annealing_warmup',
        'warmup_steps': 10000,
        'max_steps': 1000000,
        'min_learning_rate': 1e-6,
        'cycle_mult': 1.0
    },

    'regularization': {
        'label_smoothing': 0.1,
        'gradient_clip': 1.0,
        'dropout_variational': True,
        'stochastic_depth': 0.1,
        'weight_standardization': True
    }
}

# 实验跟踪与评估配置
EXPERIMENT_TRACKING_CONFIG = {
    'metrics_tracking': {
        'primary_metrics': ['accuracy', 'f1_score', 'auroc', 'auprc'],
        'secondary_metrics': ['precision', 'recall', 'specificity'],
        'uncertainty_metrics': ['expected_calibration_error', 'confidence_accuracy'],
        'training_metrics': ['loss', 'learning_rate', 'gradient_norm']
    },

    'model_checkpoints': {
        'checkpoint_strategy': 'best_k',
        'save_top_k': 3,
        'monitor_metric': 'val_auroc',
        'save_frequency': 1000,
        'keep_checkpoint_every_n_hours': 2
    },

    'hyperparameter_tuning': {
        'tuning_strategy': 'bayesian_optimization',
        'max_trials': 100,
        'parallel_trials': 8,
        'early_stopping_patience': 10
    }
}

# 高级特征工程配置
ADVANCED_FEATURE_ENGINEERING = {
    'evolutionary_features': {
        'msa_generation': {
            'database': 'uniref90',
            'e_value': 1e-3,
            'coverage': 0.8,
            'max_sequences': 1000
        },
        'profile_features': ['position_frequency', 'position_weight', 'entropy'],
        'contact_prediction': 'plm_probability'
    },

    'geometric_features': {
        'distance_metrics': ['euclidean', 'manhattan', 'cosine'],
        'angle_features': ['dihedral', 'bond_angles', 'torsion'],
        'surface_features': ['solvent_accessibility', 'curvature', 'shape_index']
    },

    'temporal_features': {
        'dynamics_simulation': {
            'time_steps': 1000,
            'sampling_frequency': 10,
            'feature_extraction': ['fluctuation', 'correlation', 'autocorrelation']
        }
    }
}

# 外部工具集成配置
EXTERNAL_TOOL_INTEGRATION = {
    'structure_prediction': {
        'alphafold_integration': {
            'model_type': 'alphafold2',
            'confidence_threshold': 0.7,
            'use_templates': True,
            'relax_structure': True
        }
    },

    'molecular_dynamics': {
        'simulation_engine': 'openmm',
        'force_field': 'amber14',
        'simulation_time': '100ns',
        'trajectory_analysis': True
    },

    'database_connections': {
        'pdb_database': {
            'host': 'rcsb.org',
            'cache_enabled': True,
            'prefetch_size': 1000
        },
        'uniprot_database': {
            'host': 'uniprot.org',
            'batch_size': 100,
            'timeout': 30
        }
    }
}

# 部署与服务化配置
DEPLOYMENT_CONFIG = {
    'model_serving': {
        'framework': 'triton_inference',
        'max_batch_size': 32,
        'gpu_memory_fraction': 0.8,
        'dynamic_batching': True
    },

    'api_endpoints': {
        'prediction_endpoint': '/v1/predict',
        'feature_extraction_endpoint': '/v1/features',
        'uncertainty_endpoint': '/v1/uncertainty',
        'batch_processing_endpoint': '/v1/batch'
    },

    'monitoring': {
        'performance_metrics': ['throughput', 'latency', 'error_rate'],
        'resource_metrics': ['gpu_utilization', 'memory_usage', 'cpu_usage'],
        'alerting_thresholds': {
            'max_latency_ms': 1000,
            'max_error_rate': 0.01,
            'min_throughput': 10
        }
    }
}