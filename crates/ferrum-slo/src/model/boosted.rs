use super::{BasePrediction, ModelError, ScaledSample, TrainingBudget, WorkCounter, FEATURE_COUNT};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Node {
    value: f64,
    split: Option<Split>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Split {
    feature: usize,
    threshold: f64,
    left: usize,
    right: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tree {
    #[serde(deserialize_with = "deserialize_nodes")]
    nodes: Vec<Node>,
}

impl Tree {
    fn predict(&self, x: &[f64; FEATURE_COUNT]) -> f64 {
        let mut index = 0;
        loop {
            let node = &self.nodes[index];
            match &node.split {
                Some(split) => {
                    index = if x[split.feature] <= split.threshold {
                        split.left
                    } else {
                        split.right
                    }
                }
                None => return node.value,
            }
        }
    }

    fn validate(&self, budget: &TrainingBudget) -> Result<(), ModelError> {
        if self.nodes.is_empty() || self.nodes.len() > budget.nodes_per_tree() {
            return Err(ModelError::ResourceLimit("persisted tree nodes"));
        }
        // Every node must be reached exactly once from root, with bounded depth.
        let mut visited = [false; 15];
        self.visit(0, 0, budget.max_depth, &mut visited)?;
        if visited[..self.nodes.len()].iter().any(|x| !x) {
            return Err(ModelError::InvalidData("unreachable tree node"));
        }
        Ok(())
    }

    fn visit(
        &self,
        index: usize,
        depth: usize,
        max_depth: usize,
        visited: &mut [bool; 15],
    ) -> Result<(), ModelError> {
        if depth > max_depth || index >= self.nodes.len() || visited[index] {
            return Err(ModelError::InvalidData("tree cycle, index or depth"));
        }
        visited[index] = true;
        let node = &self.nodes[index];
        if !node.value.is_finite() || node.value.abs() > 86400.0 {
            return Err(ModelError::InvalidData("nonfinite or unbounded tree value"));
        }
        if let Some(split) = &node.split {
            if split.feature >= FEATURE_COUNT
                || !split.threshold.is_finite()
                || !(0.0..=1.0).contains(&split.threshold)
            {
                return Err(ModelError::InvalidData("invalid tree split"));
            }
            self.visit(split.left, depth + 1, max_depth, visited)?;
            self.visit(split.right, depth + 1, max_depth, visited)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Head {
    initial: f64,
    #[serde(deserialize_with = "deserialize_trees")]
    trees: Vec<Tree>,
}

fn deserialize_nodes<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Node>, D::Error> {
    deserialize_bounded::<D, Node, 15>(deserializer)
}

fn deserialize_trees<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Tree>, D::Error> {
    deserialize_bounded::<D, Tree, 32>(deserializer)
}

fn deserialize_bounded<'de, D, T, const MAX: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Visitor<T, const MAX: usize>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const MAX: usize> serde::de::Visitor<'de> for Visitor<T, MAX> {
        type Value = Vec<T>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(formatter, "at most {MAX} model entries")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            if sequence.size_hint().is_some_and(|n| n > MAX) {
                return Err(serde::de::Error::custom("model collection limit exceeded"));
            }
            let mut entries = Vec::with_capacity(MAX);
            while entries.len() < MAX {
                match sequence.next_element()? {
                    Some(entry) => entries.push(entry),
                    None => return Ok(entries),
                }
            }
            if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                return Err(serde::de::Error::custom("model collection limit exceeded"));
            }
            Ok(entries)
        }
    }
    deserializer.deserialize_seq(Visitor::<T, MAX>(std::marker::PhantomData))
}

impl Head {
    fn predict(&self, x: &[f64; FEATURE_COUNT], rate: f64) -> f64 {
        (self.initial + rate * self.trees.iter().map(|tree| tree.predict(x)).sum::<f64>()).max(0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Boosted {
    center: Head,
    risk: Head,
    learning_rate: f64,
    pub(super) quantile: f64,
}

impl Boosted {
    pub(super) fn predict(&self, x: &[f64; FEATURE_COUNT]) -> BasePrediction {
        BasePrediction {
            central_seconds: self.center.predict(x, self.learning_rate),
            risk_base_seconds: self.risk.predict(x, self.learning_rate),
        }
    }

    pub(super) fn validate(&self, budget: &TrainingBudget) -> Result<(), ModelError> {
        if !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.quantile.is_finite()
            || self.quantile <= 0.0
            || self.quantile >= 1.0
        {
            return Err(ModelError::InvalidData("invalid boosting objective"));
        }
        for head in [&self.center, &self.risk] {
            if !head.initial.is_finite() || !(0.0..=86400.0).contains(&head.initial) {
                return Err(ModelError::InvalidData("invalid boosting base value"));
            }
            if head.trees.len() > budget.max_trees {
                return Err(ModelError::ResourceLimit("persisted trees"));
            }
            for tree in &head.trees {
                tree.validate(budget)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Loss {
    Squared,
    Pinball(f64),
}

impl Loss {
    fn gradient(self, residual: f64) -> f64 {
        match self {
            Self::Squared => residual,
            Self::Pinball(q) => {
                if residual < 0.0 {
                    q - 1.0
                } else if residual > 0.0 {
                    q
                } else {
                    0.0
                }
            }
        }
    }

    fn value(self, residual: f64) -> f64 {
        match self {
            Self::Squared => residual * residual,
            Self::Pinball(q) => {
                if residual >= 0.0 {
                    q * residual
                } else {
                    (q - 1.0) * residual
                }
            }
        }
    }

    fn optimum(self, values: &mut [f64], work: &mut WorkCounter) -> Result<f64, ModelError> {
        work.charge(values.len())?;
        match self {
            Self::Squared => Ok(values.iter().sum::<f64>() / values.len() as f64),
            Self::Pinball(q) => {
                work.charge(values.len() * (values.len().ilog2() as usize + 1))?;
                values.sort_unstable_by(f64::total_cmp);
                let index = ((q * values.len() as f64).ceil() as usize)
                    .saturating_sub(1)
                    .min(values.len() - 1);
                Ok(values[index])
            }
        }
    }
}

pub(super) fn train(
    training: &[ScaledSample],
    validation: &[ScaledSample],
    budget: &TrainingBudget,
    work: &mut WorkCounter,
) -> Result<Boosted, ModelError> {
    Ok(Boosted {
        center: train_head(training, validation, Loss::Squared, budget, work)?,
        risk: train_head(
            training,
            validation,
            Loss::Pinball(budget.risk_quantile),
            budget,
            work,
        )?,
        learning_rate: budget.learning_rate,
        quantile: budget.risk_quantile,
    })
}

fn train_head(
    training: &[ScaledSample],
    validation: &[ScaledSample],
    loss: Loss,
    budget: &TrainingBudget,
    work: &mut WorkCounter,
) -> Result<Head, ModelError> {
    let mut residual: Vec<_> = training.iter().map(|s| s.y).collect();
    let initial = loss.optimum(&mut residual, work)?;
    let mut head = Head {
        initial,
        trees: Vec::with_capacity(budget.max_trees),
    };
    let mut predictions = vec![initial; training.len()];
    let mut validation_predictions = vec![initial; validation.len()];
    let mut gradients = vec![0.0; training.len()];
    let mut best_count = 0;
    let mut best_loss = validation
        .iter()
        .map(|s| loss.value(s.y - initial))
        .sum::<f64>();
    for _ in 0..budget.max_trees {
        work.charge(training.len())?;
        for i in 0..training.len() {
            residual[i] = training[i].y - predictions[i];
            gradients[i] = loss.gradient(residual[i]);
        }
        let mut tree = Tree {
            nodes: Vec::with_capacity(budget.nodes_per_tree()),
        };
        let indices: Vec<_> = (0..training.len()).collect();
        TreeBuilder {
            training,
            residual: &residual,
            gradients: &gradients,
            loss,
            budget,
            work,
        }
        .build(&indices, 0, &mut tree)?;
        work.charge((training.len() + validation.len()) * (budget.max_depth + 1))?;
        for (sample, prediction) in training.iter().zip(&mut predictions) {
            *prediction += budget.learning_rate * tree.predict(&sample.x);
        }
        let mut validation_loss = 0.0;
        for (sample, prediction) in validation.iter().zip(&mut validation_predictions) {
            *prediction += budget.learning_rate * tree.predict(&sample.x);
            validation_loss += loss.value(sample.y - prediction.max(0.0));
        }
        head.trees.push(tree);
        if validation_loss < best_loss {
            best_loss = validation_loss;
            best_count = head.trees.len();
        }
    }
    // Each head's tree count is selected only by its own validation objective.
    head.trees.truncate(best_count);
    Ok(head)
}

struct TreeBuilder<'a> {
    training: &'a [ScaledSample],
    residual: &'a [f64],
    gradients: &'a [f64],
    loss: Loss,
    budget: &'a TrainingBudget,
    work: &'a mut WorkCounter,
}

impl TreeBuilder<'_> {
    fn build(
        &mut self,
        indices: &[usize],
        depth: usize,
        tree: &mut Tree,
    ) -> Result<usize, ModelError> {
        self.work.charge(indices.len())?;
        let mut leaf_residuals: Vec<_> = indices.iter().map(|&i| self.residual[i]).collect();
        let value = self.loss.optimum(&mut leaf_residuals, self.work)?;
        drop(leaf_residuals);
        let index = tree.nodes.len();
        tree.nodes.push(Node { value, split: None });
        if depth == self.budget.max_depth || indices.len() < 2 {
            return Ok(index);
        }
        let mut best = None;
        let total: f64 = indices.iter().map(|&i| self.gradients[i]).sum();
        let unsplit_score = total * total / indices.len() as f64;
        let mut best_gain = 1e-15 * (1.0 + unsplit_score.abs());
        for feature in 0..FEATURE_COUNT {
            self.work
                .charge(indices.len() * (indices.len().ilog2() as usize + 3))?;
            let mut ordered: Vec<_> = indices
                .iter()
                .map(|&i| (self.training[i].x[feature], self.gradients[i]))
                .collect();
            ordered.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            let mut left_sum = 0.0;
            let mut candidate = 1;
            for left_count in 1..ordered.len() {
                left_sum += ordered[left_count - 1].1;
                if left_count * (self.budget.split_candidates + 1) < candidate * ordered.len() {
                    continue;
                }
                candidate += 1;
                if candidate > self.budget.split_candidates + 1 {
                    break;
                }
                if ordered[left_count - 1].0 == ordered[left_count].0 {
                    continue;
                }
                let right_sum = total - left_sum;
                let gain = left_sum * left_sum / left_count as f64
                    + right_sum * right_sum / (ordered.len() - left_count) as f64
                    - unsplit_score;
                if gain > best_gain {
                    best_gain = gain;
                    best = Some((feature, ordered[left_count - 1].0));
                }
            }
        }
        if let Some((feature, threshold)) = best {
            self.work.charge(indices.len())?;
            let mut left = Vec::with_capacity(indices.len());
            let mut right = Vec::with_capacity(indices.len());
            for &i in indices {
                if self.training[i].x[feature] <= threshold {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }
            let left_index = self.build(&left, depth + 1, tree)?;
            let right_index = self.build(&right, depth + 1, tree)?;
            tree.nodes[index].split = Some(Split {
                feature,
                threshold,
                left: left_index,
                right: right_index,
            });
        }
        Ok(index)
    }
}
