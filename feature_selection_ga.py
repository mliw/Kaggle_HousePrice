from deap import base, creator
import random
from deap import tools
import numpy as np


def nominal_generator(p):
    return np.random.binomial(1,p,1)[0]


class FeatureSelectionGA:
    """
        FeaturesSelectionGA
        This class uses Genetic Algorithm to find out the best features for an input model
        using Distributed Evolutionary Algorithms in Python(DEAP) package. Default toolbox is
        used for GA but it can be changed accordingly.

    
    """
    def __init__(self,model,data_cache,length_of_features,evaluate_function=None,probability=0.5):
        
        """
        Parameters
        ----------
        model : Any estimator which has predict and fit method
            The benchmark model to evaluate a certain subset of features.
        data_cache : A dict or list. 
            The data used to evaluate features.
        length_of_features : int
            The length of your features. For example, if you have 200 features, then set this number to 200.
        evaluate_function : A function whose kwargs is "model", "data_cache" and "selected_features". "selected_features"
            is a vector consisted of 0 and 1. This vector represents a subset of total features. This function returns a score
            to demonstrate the efficiency of the model. Please note, this program trys to maximize this score, so please multiply
            -1 if the return of evaluate_function is the error of your model!
        """
        
        print("""Please note! this program trys to maximize the score of evaluate_function, so please multiply -1 if the return of evaluate_function is the error of your model!""")
        
        self.model = model
        self.data_cache = data_cache
        self.n_features = length_of_features
        self.evaluate_function = evaluate_function
        self.toolbox = None
        self.creator = self._create()
        
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None
        self.pop = None
        self.best_generations = []
        self.best_pop_each_generation = None
        self.pop_fresh = True
        self.probability = probability
        print("new")
    
    def evaluate(self,individual):
        
        return self.evaluate_function(self.model,self.data_cache,individual),
    
    
    def _create(self):
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator
    
    def create_toolbox(self):
        """ 
            Custom creation of toolbox.
            Parameters
            -----------
                self
            Returns
            --------
                Initialized toolbox
        """
        
        self._init_toolbox()
        return toolbox
        
    def register_toolbox(self,toolbox):
        """ 
            Register custom created toolbox. Evalute function will be registerd
            in this method.
            Parameters
            -----------
                Registered toolbox with crossover,mutate,select tools except evaluate
            Returns
            --------
                self
        """
        toolbox.register("evaluate", self.evaluate)
        self.toolbox = toolbox
     
    
    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", nominal_generator,self.probability)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox
        
        
    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=8)
        toolbox.register("evaluate", self.evaluate)
        return toolbox
    
    def get_final_scores(self,pop,fits):
        self.final_fitness = list(zip(pop,fits))
        
         
    def generate(self,n_pop,cxpb = 0.5,mutxpb = 0.2,ngen=5,set_toolbox = False):
        
        """ 
            Generate evolved population
            Parameters
            -----------
                n_pop : {int}
                        population size
                cxpb  : {float}
                        crossover probablity
                mutxpb: {float}
                        mutation probablity
                n_gen : {int}
                        number of generations
                set_toolbox : {boolean}
                              If True then you have to create custom toolbox before calling 
                              method. If False use default toolbox.
            Returns
            --------
                Fittest population
        """
        
        if not set_toolbox:
            self.toolbox = self._default_toolbox()
        else:
            raise Exception("Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use defualt toolbox")
        if self.pop_fresh:
            self.pop = self.toolbox.population(n_pop)
            self.best_pop_each_generation = []
            self.pop_fresh = False
        CXPB, MUTPB, NGEN = cxpb,mutxpb,ngen

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("-- GENERATION {} --".format(g+1))
            offspring = self.toolbox.select(self.pop, len(self.pop))
            self.fitness_in_generation[str(g+1)] = max([ind.fitness.values[0] for ind in self.pop])
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, weak_ind))
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness.values = fit
            print("Evaluated %i individuals" % len(weak_ind))

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring
            
                    # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in self.pop]
            
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
    
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            
            tem_best = tools.selBest(self.pop, 5)[0]
            self.best_generations.append(tools.selBest(self.pop, 1)[0])
            #for tem_tem_best in tem_best:
            #    self.best_pop_each_generation.append([tem_tem_best,tem_tem_best.fitness.values])
        print("-- Only the fittest survives --")

        self.best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        self.get_final_scores(self.pop,fits)
        

    
   
    
    
