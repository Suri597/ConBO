# Complex coupling of Requirements, sub components and compute_robustness

# class Requirements:
#     1) holds sub specs, mapping, _call_ to compute robustness
#     2) What specs are active
#     3) holds y_train for components
#     4) should be able to give out emi-EI
#     5) classification inside or outside?
from abc import ABC
from typing import Sequence, Dict, Set
from numpy.typing import NDArray

import numpy as np
from copy import deepcopy
from staliro.specifications import RTAMTDense
from staliro.core import Specification
from scipy.stats import norm
from scipy import stats
from scipy.optimize import minimize
# from pyswarms.single.global_best import GlobalBestPSO
import time

from ..classifierInterface import Classifier
from ..sampling import lhs_sampling, uniform_sampling
from ..utils import sample_spec
from ..gprInterface import GPR
from .specGPR import SpecEI

class Component:
    def __init__(self, identifier, spec, pred_mapping, mapping) -> None:
        self.id = identifier
        self.spec = spec
        self.pred_mapping = pred_mapping
        self.specification = RTAMTDense(self.spec, self.pred_mapping)
        self.count = 0
        self.robustness_history = []
        self.falsified = False
        self.io_mapping = mapping
        self.monitoring_time = []

    def __call__(self, states, times):
        self.count += 1
        start_time = time.perf_counter()
        robustness = self.specification.evaluate(states, times)
        self.monitoring_time.append(time.perf_counter() - start_time)
        
        self.robustness_history.append([self.count, robustness])
        if robustness < 0:
            self.falsified = True
        return robustness

    


class Requirement(Specification[Sequence[float], float], ABC):
    def __init__(self, tf_dim, component_list, predicate_mapping) -> None:
        self.tf_dim = tf_dim
        self.component_list = component_list
        self.predicate_mapping = predicate_mapping
        self.requirements = []
        # self.falsified_reqs = []
        self.overall_count = 0
        # self.num_components = len(component_list)
        
        for iter, spec in enumerate(component_list):
            predicate_mapping_local = {}
            input_indices_component = []
            for var in predicate_mapping.keys():
                if var in spec:
                    index, predicate_mapping_local[var] = predicate_mapping[var]
                    if index not in input_indices_component:
                        input_indices_component += index
                    # input_indices.update(index)
            mapping = np.array([1 if item in input_indices_component else 0 for item in range(tf_dim)])
            
            
            self.requirements.append(Component(iter, spec, predicate_mapping_local, mapping))   
    
    def evaluate(self, states: NDArray[np.float_], times: NDArray[np.float_]) -> Dict[int, float]:
        self.overall_count += 1
        component_rob = {}
        
        num_requirements = len(self.requirements)

        for iterate in range(num_requirements):
            if not self.requirements[iterate].falsified:
                result = self.requirements[iterate](states, times)
                component_rob[self.requirements[iterate].id] = result
                
        return component_rob

    def __len__(self) -> int:
        return len(self.requirements)

    @property
    def falsified_components(self) -> Set[int]:
        return {req.id for req in self.requirements if req.falsified}

    @property
    def unfalsified_components(self) -> Set[int]:
        return {req.id for req in self.requirements if not req.falsified}

    @property
    def num_falsified_components(self):
        return len(self.falsified_components)

    @property
    def num_unfalsified_components(self):
        return len(self.unfalsified_components)
    
    @property
    def specification_reset(self):
        tf_dim = self.tf_dim
        component_list = self.component_list
        predicate_mapping = self.predicate_mapping
        self.requirements = []
        # self.falsified_reqs = []
        self.overall_count = 0
        # self.num_components = len(component_list)
        
        for iter, spec in enumerate(component_list):
            predicate_mapping_local = {}
            input_indices_component = []
            for var in predicate_mapping.keys():
                if var in spec:
                    index, predicate_mapping_local[var] = predicate_mapping[var]
                    if index not in input_indices_component:
                        input_indices_component += index
                    # input_indices.update(index)
            mapping = np.array([1 if item in input_indices_component else 0 for item in range(tf_dim)])
            
            
            self.requirements.append(Component(iter, spec, predicate_mapping_local, mapping))

    def _get_complete_data(self):
        
        y_new = {}
        for iterate in range(len(self.requirements)):
            y_new[iterate] = np.array(self.requirements[iterate].robustness_history)[:,-1]

        return y_new


    def _get_unfalsified_data(self):
        y_train_active_comp = self.unfalsified_components
        y_new = {}
        
        for iterate in y_train_active_comp:
            
            y_new[iterate] = np.array(self.requirements[iterate].robustness_history)[:,-1]

        return y_new

    def _generate_unfaslified_dataset(self):
        
        robs_data = self._get_unfalsified_data()
        idxs = list(robs_data.keys())
        data = np.array(list(robs_data.values())).T

        return idxs, data

    def _get_individual_monitoring_times(self):
        indi_monitoring_times = {}
        for req in self.requirements:
            indi_monitoring_times[req.id] = req.monitoring_time
        
        return indi_monitoring_times
        