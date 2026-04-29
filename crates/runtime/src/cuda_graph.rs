use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaGraphBucket {
    pub max_context: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaGraphPlan {
    pub buckets: Vec<CudaGraphBucket>,
}

impl Default for CudaGraphPlan {
    fn default() -> Self {
        Self {
            buckets: [1024, 4096, 16384, 65536, 262144]
                .into_iter()
                .map(|max_context| CudaGraphBucket { max_context })
                .collect(),
        }
    }
}

impl CudaGraphPlan {
    pub fn bucket_for(&self, context: usize) -> Option<CudaGraphBucket> {
        self.buckets
            .iter()
            .copied()
            .find(|bucket| context <= bucket.max_context)
    }
}

