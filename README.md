# DEQ-AD

## Deep Equilibrium Network for Alzheimer's Disease Classification
### and Cognitive Severity Prediction from Structural MRI

DEQ-AD is a research implementation of an equilibrium-based deep
learning framework for subject-level Alzheimer's disease classification
from structural MRI.

The framework combines a lightweight 3D MRI encoder with a Deep
Equilibrium (DEQ) layer and evaluates the resulting equilibrium
representation for binary cognitive impairment classification.

---

## Research Question

Can an equilibrium-based neural representation learn clinically
meaningful patterns from structural MRI for Alzheimer's disease
classification and cognitive-severity prediction?

---

## Architecture

```text
3D Structural MRI
       |
       v
  3D CNN Encoder
       |
       v
  64-D Feature Vector
       |
       v
 Deep Equilibrium Layer
       |
       v
 Equilibrium Representation
      / \
     /   \
    v     v
Binary   CDR
Head     Head
