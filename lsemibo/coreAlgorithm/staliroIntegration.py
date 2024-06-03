from dataclasses import dataclass
from typing import Any, List, Sequence, Callable
from ..gprInterface import GPR
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import comb
from scipy.stats import norm
import numpy as np
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample
from attr import frozen

# from .lsemibo import LSemiBO
from ..sampling import uniform_sampling, lhs_sampling
from ..utils import sample_spec, sample_spec_gp
from .specGPR import SpecEI, minSpecEI
from ..classifierInterface import Classifier
import pyswarms as ps

from copy import deepcopy
from itertools import combinations
import numpy as np
from tqdm import tqdm
import time
import math
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, Any
from attr import frozen

Bounds = Sequence[Interval]

@frozen(slots=True)
class LSemiBOResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
    """

    samples: Any
    components: Any
    total_time: Any
    history: Any
    individual_monitoring_time: Any
    sample_generation_time: Any
    topk_time: Any

@dataclass(frozen=False)
class LSemiBOOptimizer(Optimizer[float, LSemiBOResult]):
    """The LSemiBO optimizer provides falsifying inputs in a conjunctive requirement scenario."""

    method: str
    is_budget: int
    max_budget: int
    cs_budget: int
    top_k: int
    classified_sample_bias: float
    tf_dim: int
    R: int
    M: int
    gpr_model: Callable
    classifier_model: Callable
    is_type: str
    cs_type: str
    pi_type: str
    seed: int
    

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget:int, seed: int) -> LSemiBOResult:
        self._set_budgets(budget)
        self._set_func_and_num_requirements(func)
        self._set_region_support(bounds)
        self._set_rng(seed)

        samples,components, total_time, history,  individual_monitoring_time,sample_generation_time, topk_time = self._sample()
        return LSemiBOResult(samples,components, total_time, history, individual_monitoring_time,sample_generation_time, topk_time)

    def _set_rng(self, seed):
        self.seed = self.seed
        self.rng = np.random.default_rng(seed)
    
    def _set_func_and_num_requirements(self, func):
        func.specification.specification_reset 
        self.func = func
        self.num_requirements = len(self.func.specification)
    
    def _set_budgets(self, max_budget):
        #self.max_budget = self.max_budget
        self.bo_budget = self.max_budget - self.is_budget
        self.rem_budget = self.max_budget - self.is_budget
    
    def _set_region_support(self, bounds):
        self.region_support = np.array((tuple(bound.astuple() for bound in bounds),))[0]

    def _sample(self):
        # time_stats = {}
                    

        total_start_time = time.perf_counter()
        print(f"Starting Replication for seed {self.seed}")

        if self.is_type == "lhs_sampling":
            x_train = lhs_sampling(self.is_budget, self.region_support, self.tf_dim, self.seed)
        elif self.is_type == "uniform_sampling":
            x_train = uniform_sampling(self.is_budget, self.region_support, self.tf_dim, self.seed)
        else:
            raise ValueError(f"{self.is_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        # print(y_train)
        #print("x_train_before",x_train)
        #for l in x_train:
            #print("mmmm",l)
        x_train_ = [Sample(tuple(l)) for l in x_train]
        y_train = self.func.eval_samples(x_train_)
        #print("x_train_after",x_train_)
       
        # y_train = compute_robustness(x_train, self.specification, self.tf_wrapper)
        
        
        topk_time = {}
        sample_generation_time = {}
        pred_mean_Y = {}
        x_candidate = {}
        optimal_pair_set = np.zeros((self.max_budget - x_train.shape[0],2))

        #print(f"x_train.shape[0]*******************{x_train.shape[0]}")
        #print(f"self.max_budget********************{self.max_budget}")
        for budget in tqdm(range(self.max_budget - x_train.shape[0])):
            
            # print(f"***************************************************")
            # print(f"********************{budget}***********************")
            print(f"Falsified Components: {self.func.specification.num_falsified_components}\n Unfalsified components remaining: {self.func.specification.num_unfalsified_components}")
            if self.func.specification.num_unfalsified_components <= self.num_requirements:
                c1 = (self.func.specification.num_unfalsified_components == 0 and self.method == "falsification_elimination")
                c2 = (self.func.specification.num_unfalsified_components < self.num_requirements and self.method == "falsification")
                if c1 or c2:
                    # time_stats["total_time"] = time.perf_counter() - total_start_time
                    # time_stats["monitoring_time"] = self.tf_wrapper.monitoring_time
                    # time_stats["simulation_time"] = self.tf_wrapper.sim_time
                    # time_stats["individual_monitoring_time"] = self.requirement._get_individual_monitoring_times()
                    # time_stats["sample_generation_time"] = sample_generation_time
                    # time_stats["topk_time"] = topk_time



                    print("All reqs falsified")
                    print(f"Ending Replication Early for seed {self.seed}")
                    if c1:
                        print("**************************************")
                        print("*******************Date Saved*******************")
                        print("**************************************")
                        #with open(f'NLF_conBOLS_attribute_seed_{budget}', 'wb') as file:
                            #pickle.dump(x_candidate,pred_mean_Y, file)
                        print("Ending early due to elimination of all reqs")
                    elif c2:
                        print("**************************************")
                        print("*******************Date Saved*******************")
                        print("**************************************")
                        #with open(f'NLF_conBOLS_attribute_seed_{budget}', 'wb') as file:
                            #pickle.dump(x_candidate,pred_mean_Y, file)
                        print("Ending early due to falsification")
                    # output_data = {}
                    # output_data["samples"] = x_train
                    # output_data["components"] = self.requirement._get_complete_data()
                    # output_data["time_res"] = time_stats
                    # print(f"Ending Replication After Exhuasting budget for Run {self.run_number} with seed {self.seed}")
                    # with open(self.benchmark_directory.joinpath(self.benchmark_name + f"_{self.method}_point_history{self.run_number}.pkl"), "wb") as f:
                    #     pickle.dump(output_data, f)
                        
                    #with open(f'NLF_conBOLS_x_candidate_{self.top_k}seed_{self.seed}', 'wb') as file:
                        #pickle.dump(x_candidate, file) 
                    #with open(f'NLF_conBOLS_pred_mean_Y_seed_{self.seed}', 'wb') as file:
                        #pickle.dump(pred_mean_Y, file) 
                    #with open(f'NLF_conBOLS_pair_pattern_{self.top_k}_seed_{self.seed}', 'wb') as file:
                        #pickle.dump(optimal_pair_set, file) 

                    return x_train, \
                        self.func.specification._get_complete_data(), \
                        time.perf_counter() - total_start_time,\
                        self.func.history,\
                        self.func.specification._get_individual_monitoring_times(),\
                        sample_generation_time,\
                        topk_time
                            
                else:
                    if self.func.specification.num_unfalsified_components == 1:
                        #pred_sample_x, candidate_X, candidate_EI,topk_time_sample, sample_generation_time_sample = self.minbo_req(x_train, self.gpr_model, self.classifier_model, self.seed+budget+5000)
                        pred_sample_x, candidate_X, candidate_EI,topk_time_sample, sample_generation_time_sample = self.minbo_req(x_train, self.gpr_model, self.classifier_model, self.rng)
                        sample_generation_time[self.is_budget + budget+1] = sample_generation_time_sample
                        x_candidate[budget] = candidate_X
                        pred_mean_Y[budget+3] = candidate_EI
                        x_train = np.vstack((x_train, np.array([pred_sample_x])))
                        pred_sample_y = self.func.eval_sample(Sample(tuple(pred_sample_x)))
                      
                    
                    else:
                    
                        #pred_sample_x, candidate_X, Pred_Y, topk_time_sample, sample_generation_time_sample, optimal_pair = self.lsemibo_req(x_train, self.gpr_model, self.classifier_model, self.seed+budget+5000)
                        pred_sample_x, candidate_X, Pred_Y, topk_time_sample, sample_generation_time_sample, optimal_pair = self.lsemibo_req(x_train, self.gpr_model, self.classifier_model, self.rng)
                        optimal_pair_set[budget, :] =  optimal_pair
                        #print("optimal_pair_set",optimal_pair_set)
                        topk_time[self.is_budget + budget+1] = topk_time_sample
                        sample_generation_time[self.is_budget + budget+1] = sample_generation_time_sample
                        x_candidate[budget] = candidate_X
                        pred_mean_Y[budget+3] = Pred_Y
                        x_train = np.vstack((x_train, np.array([pred_sample_x])))
                        pred_sample_y = self.func.eval_sample(Sample(tuple(pred_sample_x)))
                        
                    
                    
        print(f"Ending Replication After Exhuasting budget for seed {self.seed}")
       

        return x_train, \
                self.func.specification._get_complete_data(), \
                time.perf_counter() - total_start_time,\
                self.func.history,\
                self.func.specification._get_individual_monitoring_times(),\
                sample_generation_time,\
                topk_time

    def lsemibo_req(self, x_train, gpr_model, classifier_model, rng):
    #def lsemibo_req(self, x_train, gpr_model, classifier_model, seed):
        
        #Obtaining true function value
        idxs, y_train = self.func.specification._generate_unfaslified_dataset()
        idxs_dict = dict(zip(idxs, list(range(len(idxs)))))
        best_point = np.min(y_train)
        #print("best_point1",best_point)
        #Gaussian Process training
        gpr_model_dict = {}

        for i in idxs_dict:
            gpr_model_dict[i] = (deepcopy(GPR(gpr_model)))
            gpr_model_dict[i].fit(x_train, y_train[:, idxs_dict[i]])

        #Converting data for classifier
        

        t_start_topk_time = time.perf_counter()
        

        sampled_specs = self.choose_top_kgp(best_point, idxs, gpr_model_dict, self.rng)
        #sampled_specs = self.choose_top_kgp(best_point, idxs, gpr_model_dict, seed)

        y_train_classes = np.argmin(y_train, 1)        
        #sampled_specs = self.choose_top_k(x_train, y_train_classes, idxs, classifier_model, seed)
        #sampled_specs = self.choose_top_k(x_train, y_train_classes, idxs, classifier_model, self.rng)
        
       
        print(f"sampled_spec:{sampled_specs}")
        topk_time = time.perf_counter() - t_start_topk_time

        
        #component_ei = []
        
        print("*************************************************")
        print(f"best function value achieved:  {best_point}")
        print("*************************************************")
        #pi = []
        pair_ind = list(combinations(range(len(sampled_specs)),2))
        
        pair_idxs = np.zeros((comb(len(sampled_specs),2),2))

        #print("*************************************************")
        #print(f"Active Requirement Pairs:{pair_ind}")
        #print("*************************************************")
        
        component_pred_y = []
        component_x = []
       


        
        #print("gpr_model_dict",gpr_model_dict)

        t_start_sample_gen_time = time.perf_counter()
        for i, iterate in enumerate(pair_ind):
            
            mapping_indices_1 = np.where(self.func.specification.requirements[iterate[0]].io_mapping == 1)[0]
            mapping_indices_2 = np.where(self.func.specification.requirements[iterate[1]].io_mapping == 1)[0]
            
            component1 = iterate[0]
            component2 = iterate[1]

            pair_idxs[i,0] = list(sampled_specs)[component1]
            pair_idxs[i,1] = list(sampled_specs)[component2]

            print(list(sampled_specs)[component1], list(sampled_specs)[component2])

            gpr_model1 = gpr_model_dict[list(sampled_specs)[component1]]
            gpr_model2 = gpr_model_dict[list(sampled_specs)[component2]]

            y_train_subset1 = y_train[:, idxs_dict[list(sampled_specs)[component1]]]
            y_train_subset2 = y_train[:, idxs_dict[list(sampled_specs)[component2]]]

            req_comp = SpecEI(iterate, x_train, y_train_subset1, y_train_subset2, best_point, mapping_indices_1, mapping_indices_2, gpr_model1, gpr_model2, self.region_support, self.R, self.M, self.tf_dim, self.rng, self.pi_type)
            #p_sample_x, pred_mean = self._opt_acquisition(req_comp, self.region_support, self.tf_dim, seed, list(sampled_specs)[component1], list(sampled_specs)[component2])
            p_sample_x, pred_mean = self._opt_acquisition(req_comp, self.region_support, self.tf_dim, rng,list(sampled_specs)[component1], list(sampled_specs)[component2])
            
            component_x.append(p_sample_x)
            component_pred_y.append(pred_mean)
    
        sample_idxs = np.argmin(component_pred_y)
        pred_sample_x = component_x[sample_idxs]
        optimal_pair = pair_idxs[sample_idxs,:]

  
        sample_generation_time = time.perf_counter() - t_start_sample_gen_time
        
        return pred_sample_x, component_x,component_pred_y, topk_time, sample_generation_time, optimal_pair
    
    def minbo_req(self, x_train, gpr_model, classifier_model, rng):
    #def minbo_req(self, x_train, gpr_model, classifier_model, seed):
        t_start_sample_gen_time = time.perf_counter()
        idxs, y_train = self.func.specification._generate_unfaslified_dataset()
   

        idxs_dict = dict(zip(idxs, list(range(len(idxs)))))
        print(f"idxs_dict**************{idxs_dict}")
        #y_train_classes = np.argmin(y_train, 1)
        #y_train_classes = np.argmin(y_train, 1)
        t_start_topk_time = time.perf_counter()
        #sampled_specs = self.choose_top_k(x_train, y_train_classes, idxs, seed)
        
        topk_time = time.perf_counter() - t_start_topk_time

        sampled_specs = list(idxs_dict.keys())
        print(f"sampled_specs**************{sampled_specs}")

        component_ei = []
        component_x = []
        print("*************************************************")
        #print(y_train)
        print("*************************************************")
        best_point = np.min(y_train)
        print("*************************************************")
        #print(f"best function value achieved:  {best_point}and sample number:{np.argmin(y_train)}")
        print("*************************************************")

        
        for iterate in sampled_specs:
            print(iterate)
            component1 = iterate
            mapping_indices = np.where(self.func.specification.requirements[iterate].io_mapping == 1)[0]
            
            #x_train_subset = x_train[:, mapping_indices]
            y_train_subset = y_train[:, idxs_dict[component1]]
            
            req_comp = minSpecEI(iterate, x_train, y_train_subset, best_point, mapping_indices, gpr_model, self.region_support, self.tf_dim, self.rng)
            p_sample_x, pred_sample_ei = self.min_opt_acquisition(req_comp, self.region_support, self.tf_dim, self.rng)
            #p_sample_x, pred_sample_ei = self.min_opt_acquisition(req_comp, self.region_support, self.tf_dim, seed)
            component_x.append(p_sample_x)
            component_ei.append(pred_sample_ei)
        
        
        pred_sample_x = component_x[np.argmax(component_ei)]
        #pred_sample_x = self._opt_acquisition(component_ei, self.region_support, self.tf_dim)
        sample_generation_time = time.perf_counter() - t_start_sample_gen_time
        
        return pred_sample_x, component_x,component_ei, topk_time, sample_generation_time

    #def choose_top_kgp(self, best_point, idxs, gp_dict, seed):
    def choose_top_kgp(self, best_point, idxs, gp_dict, rng):
 
     
        if self.top_k >= self.func.specification.num_unfalsified_components:
            sampled_specs = self.func.specification.unfalsified_components

        else:
            if self.cs_type == "lhs_sampling":
                    #x_test = lhs_sampling(self.cs_budget, self.region_support, self.tf_dim, seed)
                    x_test = lhs_sampling(self.cs_budget, self.region_support, self.tf_dim, self.rng)

            elif self.cs_type == "uniform_sampling":
                #x_test = uniform_sampling(self.cs_budget, self.region_support, self.tf_dim, seed)
                x_test = uniform_sampling(self.cs_budget, self.region_support, self.tf_dim, self.rng)

            else:
                raise ValueError(f"{self.cs_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")

            gp_prob_mat = np.zeros((x_test.shape[0], len(gp_dict)))

            for i in range(len(gp_dict)):

                pred_mean, pred_std = (gp_dict[idxs[i]]).predict(x_test)

                gp_prob_mat[:,i]=norm.cdf(best_point, pred_mean, pred_std)
            
            gp_avg_prob = np.mean(gp_prob_mat, axis=0)

            spec_prob = gp_avg_prob/ np.sum(gp_avg_prob)


            gp_prob_dict = dict(zip(idxs, spec_prob))

            #_sampled_specs = sample_spec_gp(gp_prob_dict, idxs, self.top_k, seed)
            _sampled_specs = sample_spec_gp(gp_prob_dict, idxs, self.top_k, rng)
            sampled_specs = _sampled_specs
                    
        return sampled_specs

    #def choose_top_k(self, x_train, y_train, idxs, classifier_model, seed):
    def choose_top_k(self, x_train, y_train, idxs, classifier_model, rng):
        
        # top_k = min(self.top_k, self.requirement._get_num_unfalsified_comp())
        if self.top_k >= self.func.specification.num_unfalsified_components:
            sampled_specs = self.func.specification.unfalsified_components
            print("Entered c1")
        
        else:
            if np.unique(y_train).shape[0] == 1:
                _sampled_specs = np.unique(y_train)
                sampled_specs = []
                
                for iter in _sampled_specs:
                    sampled_specs.append(idxs[iter])
                # print("Enter c2.1")
            else:

                classifier_model = Classifier(classifier_model)
                classifier_model.fit(x_train, y_train)
                # print(y_train)
                if self.cs_type == "lhs_sampling":
                    #x_test = lhs_sampling(self.cs_budget, self.region_support, self.tf_dim, seed)
                    x_test = lhs_sampling(self.cs_budget, self.region_support, self.tf_dim, self.rng)
                    # x_test_classifier_active = x_test 
                elif self.cs_type == "uniform_sampling":
                    #x_test = uniform_sampling(self.cs_budget, self.region_support, self.tf_dim, seed)
                    x_test = uniform_sampling(self.cs_budget, self.region_support, self.tf_dim, self.rng)
                    # x_test_classifier_active = x_test 
                else:
                    raise ValueError(f"{self.cs_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")

                y_test_classifier = classifier_model.predict(x_test)
                _unique, _counts = np.unique(y_train, return_counts=True)
                unique, counts = np.unique(y_test_classifier, return_counts=True)
                #print("**********************")
                #print("**********************")
                #print(f"Unique in y_train is \t {_unique}")
                #print(f"Counts in y_train is \t {_counts}")
                #print(f"Unique in y_test is \t {unique}")
                #print(f"Counts in y_test is \t {counts}")
                #print("**********************")
                #print("**********************")
                spec_prob = {}
                classified_spec_prob = []
                unclassified_spec_prob = []
                for spec_number in range(len(idxs)):
                    if spec_number not in set(unique):
                        spec_prob[spec_number] = 0
                        unclassified_spec_prob.append(spec_number)
                    else:
                        spec_prob[spec_number] = counts[np.where(unique == spec_number)][0] / self.cs_budget
                        classified_spec_prob.append(spec_number)
                
                # print(spec_prob, unclassified_spec_prob, classified_spec_prob)
                #_sampled_specs = sample_spec(spec_prob, unclassified_spec_prob, classified_spec_prob, self.top_k, self.classified_sample_bias, seed)
                _sampled_specs = sample_spec(spec_prob, unclassified_spec_prob, classified_spec_prob, self.top_k, self.classified_sample_bias, rng)
                # print(classified_spec_prob)
                # print(unclassified_spec_prob)
                # print(classified_spec_prob + unclassified_spec_prob)
                # _sampled_specs = self.rng.choice(classified_spec_prob+ unclassified_spec_prob, size=self.top_k, p = [self.classified_sample_bias, 1 - self.classified_sample_bias])
                sampled_specs = []
                
                for iter in _sampled_specs:
                    sampled_specs.append(idxs[iter])
                #print("Enter c.2")
                #print(_sampled_specs)
                #print(sampled_specs)
                #print("Exit")
                    
        return sampled_specs
            


        

    
    def _emi_ei(self, x, spec_ei, c1,c2, sample_type="single"):
        if x.shape[0] != 1:
            sample_type = "multiple"

        ei, pred_mean =  spec_ei._acquisition(x, c1,c2, sample_type)
      
        return ei, pred_mean
    
    def _min_ei(self, x, spec_ei, sample_type="single"):
        if x.shape[0] != 1:
            sample_type = "multiple"

     
        ei = spec_ei._acquisition(x, sample_type)
        return ei


    #def _opt_acquisition(self, spec_ei,region_support, tf_dim, seed, c1, c2):
    def _opt_acquisition(self, spec_ei,region_support, tf_dim, rng, c1, c2):
        """Get the sample points

        Args:
            X (np.array): sample points
            y (np.array): corresponding robustness values
            model ([type]): the GP models
            sbo (list): sample points to construct the robustness values
            test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
            region_support (np.array): The bounds of the region within which the sampling is to be done.
                                        Region Bounds is M x N x O where;
                                            M = number of regions;
                                            N = test_function_dimension (Dimensionality of the test function);
                                            O = Lower and Upper bound. Should be of length 2;

        Returns:
            [np.array]: the new sample points by BO
            [np.array]: sbo - new samples for resuse
        """

        lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
        upper_bound_theta = np.ndarray.flatten(region_support[:, 1])
        bounds = (lower_bound_theta, upper_bound_theta)
        
       
        fun = lambda _x: -1 * np.array(self._emi_ei(_x, spec_ei, c1, c2)[0])
     
        #random_samples = uniform_sampling(10, region_support, tf_dim, seed)
        random_samples = uniform_sampling(2000, region_support, tf_dim, self.rng)
      

        min_bo_val = -1 * np.array(self._emi_ei(
            random_samples, spec_ei,c1,c2, sample_type="multiple"
        )[0])

        min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])[0, :]
        min_bo_val = np.min(min_bo_val)

        
        for _ in range(9):
            new_params = minimize(
                fun,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                method = "L-BFGS-B",
                x0=min_bo,
            )
            # print(new_params)

            if not new_params.success:
                continue

            if min_bo is None or fun(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = fun(min_bo)

        new_params = minimize(
            fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
        
        min_bo = new_params.x

        ei_val, pred_y_min_bo = self._emi_ei(
            min_bo, spec_ei, c1,c2,sample_type="multiple"
        )
      
        return np.array(min_bo), pred_y_min_bo
    
    def min_opt_acquisition(self, spec_ei, region_support, tf_dim,rng):
        """Get the sample points

        Args:
            X (np.array): sample points
            y (np.array): corresponding robustness values
            model ([type]): the GP models
            sbo (list): sample points to construct the robustness values
            test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
            region_support (np.array): The bounds of the region within which the sampling is to be done.
                                        Region Bounds is M x N x O where;
                                            M = number of regions;
                                            N = test_function_dimension (Dimensionality of the test function);
                                            O = Lower and Upper bound. Should be of length 2;

        Returns:
            [np.array]: the new sample points by BO
            [np.array]: sbo - new samples for resuse
        """

        lower_bound_theta = np.ndarray.flatten(region_support[:, 0])
        upper_bound_theta = np.ndarray.flatten(region_support[:, 1])
        bounds=list(zip(lower_bound_theta, upper_bound_theta))
        
        # bnds = Bounds(lower_bound_theta, upper_bound_theta)
        fun = lambda _x: -1 * self._min_ei(_x, spec_ei)
        # t = time.perf_counter()
        #random_samples = uniform_sampling(10, region_support, tf_dim, seed)
        random_samples = uniform_sampling(2000, region_support, tf_dim, self.rng)
        min_bo_val = -1 * self._min_ei(
            random_samples, spec_ei, sample_type="multiple"
        )

        #print(f"minbo_val*********************{min_bo_val}")
        min_bo = np.array([random_samples[np.argmin(min_bo_val), :]])[0, :]
        min_bo_val = np.min(min_bo_val)

        
        for _ in range(9):
            new_params = minimize(
                fun,
                bounds=list(zip(lower_bound_theta, upper_bound_theta)),
                method = "L-BFGS-B",
                x0=min_bo,
            )
            # print(new_params)

            if not new_params.success:
                continue

            if min_bo is None or fun(new_params.x) < min_bo_val:
                min_bo = new_params.x
                min_bo_val = fun(min_bo)

        new_params = minimize(
            fun, bounds=list(zip(lower_bound_theta, upper_bound_theta)), x0=min_bo
        )
                
        #new_params = dual_annealing(fun, bounds, maxiter=10, no_local_search=False)
        #print(new_params)
        min_bo = new_params.x
        min_bo_ei = new_params.fun
        
        return np.array(min_bo), -min_bo_ei

    
    
