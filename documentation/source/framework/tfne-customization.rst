TFNE Architecture & Customization
=================================

Framework Architecture
----------------------

The following illustration shows the architecture of the TFNE framework beginning from v0.21.0 onwards. The architecture is shown via an entity-sequence diagram and omits many minor functions in order to emphasize the core evolutionary loop employed by TFNE as well as the most relevant functions of each aspect.

.. figure:: ../illustrations/tfne_v0.21_entity_sequence_diagram.svg
   :align: center

   Entity-Sequence Diagram Illustrating the Architecture of TFNE


--------------------------------------------------------------------------------

Customizing TFNE
----------------

One design goal of TFNE is to be modular and provide a basic prototyping platform for the implementation of similar experimental NE algorithms. TFNE therefore supports custom algorithms, encodings and environments and provides clear abstract interfaces that summarize the requirements for their implementations. While the abstract base classes extensively describe the requirements of the function that they are overwritten with, please also consider the architecture of the TFNE framework as listed above to get a top level view of the module usage in the evolutionary process.

The following 5 abstract base classes are currently present in TFNE and the 3 most important are listed in full:

.. code-block:: python

    import tfne

    # Abstract Base classes in TFNE
    tfne.algorithms.BaseNeuroevolutionAlgorithm()
    tfne.populations.BasePopulation()
    tfne.encodings.BaseEncoding()
    tfne.encodings.BaseGenome()
    tfne.environments.BaseEnvironment()


BaseNeuroevolutionAlgorithm (see `here <https://github.com/PaulPauls/Tensorflow-Neuroevolution/tree/master/tfne/algorithms/base_algorithm.py>`_)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: python

    class BaseNeuroevolutionAlgorithm(object, metaclass=ABCMeta):
        """
        Interface for TFNE compatible algorithms, which encapsulate the functionality of initialization, evaluation,
        evolution and serialization.
        """

        @abstractmethod
        def initialize_population(self, environment):
            """
            Initialize the population according to the specified NE algorithm. Adhere to potential constraints set by the
            environment.
            @param environment: one instance or multiple instances of the evaluation environment
            """
            raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement "
                                      "'initialize_population()'")

        @abstractmethod
        def evaluate_population(self, environment) -> (int, float):
            """
            Evaluate all members of the population on the supplied evaluation environment by passing each member to the
            environment and assigning the resulting fitness back to the member. Return the generation counter and the best
            achieved fitness.
            @param environment: one instance or multiple instances of the evaluation environment
            @return: tuple of generation counter and best fitness achieved by best member
            """
            raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'evaluate_population()'")

        @abstractmethod
        def summarize_population(self):
            """
            Print summary of the algorithm's population to stdout to inform about progress
            """
            raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'summarize_evaluation()'")

        @abstractmethod
        def evolve_population(self) -> bool:
            """
            Evolve all members of the population according to the NE algorithms specifications. Return a bool flag
            indicating if the population went extinct during the evolution
            @return: bool flag, indicating ig population went extinct during evolution
            """
            raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'evolve_population()'")

        @abstractmethod
        def save_state(self, save_dir_path):
            """
            Save the state of the algorithm and the current evolutionary process by serializing all aspects to json
            compatible dicts and saving it as file to the supplied save dir path.
            @param save_dir_path: string of directory path to which the state should be saved
            """
            raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'save_state()'")

        @abstractmethod
        def get_best_genome(self) -> BaseGenome:
            """
            @return: best genome so far determined by the evolutionary process
            """
            raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'get_best_genome()'")

        @abstractmethod
        def get_eval_instance_count(self) -> int:
            """
            @return: int, specifying how many evaluation threads the NE algorithm uses
            """
            raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement "
                                      "'get_eval_instance_count()'")


BaseGenome (see `here <https://github.com/PaulPauls/Tensorflow-Neuroevolution/tree/master/tfne/encodings/base_genome.py>`_)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: python

    class BaseGenome(object, metaclass=ABCMeta):
        """
        Interface for TFNE compatible genomes, which encapsulates all necessary functionality used by the algorithm,
        evaluation environment, visualizer, etc.
        """

        @abstractmethod
        def __call__(self, inputs) -> tf.Tensor:
            """
            Call genome to start inference based on the internal model. Return the results of the inference.
            @param inputs: genome model inputs
            @return: results of the genome model inference
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement '__call__()'")

        @abstractmethod
        def __str__(self) -> str:
            """
            @return: string representation of the genome
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement '__str__()'")

        @abstractmethod
        def visualize(self, show, save_dir_path, **kwargs) -> str:
            """
            Visualize the genome. If 'show' flag is set to true, display the genome after rendering. If 'save_dir_path' is
            supplied, save the rendered genome as file to that directory. Return the saved file path as string.
            @param show: bool flag, indicating whether the rendered genome should be displayed or not
            @param save_dir_path: string of the save directory path the rendered genome should be saved to.
            @param kwargs: Optional additional arguments relevant for rendering of the specific genome implementation.
            @return: string of the file path to which the rendered genome has been saved to
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'visualize()'")

        @abstractmethod
        def serialize(self) -> dict:
            """
            @return: serialized constructor variables of the genome as json compatible dict
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'serialize()'")

        @abstractmethod
        def save_genotype(self, save_dir_path) -> str:
            """
            Save genotype of genome to 'save_dir_path' directory. Return file path to which the genotype has been saved to
            as string.
            @param save_dir_path: string of the save directory path the genotype should be saved to
            @return: string of the file path to which the genotype has been saved to
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'save_genotype()'")

        @abstractmethod
        def save_model(self, file_path, **kwargs):
            """
            Save TF model of genome to specified file path.
            @param file_path: string of the file path the TF model should be saved to
            @param kwargs: Optional additional arguments relevant for TF model.save()
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'save_model()'")

        @abstractmethod
        def set_fitness(self, fitness):
            """
            Set genome fitness value to supplied parameter
            @param fitness: float of genome fitness
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'set_fitness()'")

        @abstractmethod
        def get_genotype(self) -> Any:
            """
            @return: One or multiple variables representing the genome genotype
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'get_genotype()'")

        @abstractmethod
        def get_model(self) -> tf.keras.Model:
            """
            @return: TF model represented by genome genotype
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'get_model()'")

        @abstractmethod
        def get_optimizer(self) -> Union[None, tf.keras.optimizers.Optimizer]:
            """
            Return either None or TF optimizer depending on if the genome encoding associates an optimizer with the genome
            @return: None | TF optimizer associated with genome
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'get_optimizer()'")

        @abstractmethod
        def get_id(self) -> int:
            """
            @return: int of genome ID
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'get_id()'")

        @abstractmethod
        def get_fitness(self) -> float:
            """
            @return: float of genome fitness
            """
            raise NotImplementedError("Subclass of BaseGenome does not implement 'get_fitness()'")


BaseEnvironment (see `here <https://github.com/PaulPauls/Tensorflow-Neuroevolution/tree/master/tfne/environments/base_environment.py>`_)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: python

    class BaseEnvironment(object, metaclass=ABCMeta):
        """
        Interface for TFNE compatible environments, which are supposed to encapsulate a problem and provide the necessary
        information and functions that the TFNE pre-implemented algorithms require.
        """

        @abstractmethod
        def eval_genome_fitness(self, genome) -> float:
            """
            Evaluates the genome's fitness in either the weight-training or non-weight-training variant. Returns the
            determined genome fitness.
            @param genome: TFNE compatible genome that is to be evaluated
            @return: genome calculated fitness
            """
            raise NotImplementedError("Subclass of BaseEnvironment does not implement 'eval_genome_fitness()'")

        @abstractmethod
        def replay_genome(self, genome):
            """
            Replay genome on environment by calculating its fitness and printing it.
            @param genome: TFNE compatible genome that is to be evaluated
            """
            raise NotImplementedError("Subclass of BaseEnvironment does not implement 'replay_genome()'")

        @abstractmethod
        def duplicate(self) -> BaseEnvironment:
            """
            @return: New instance of the environment with identical parameters
            """
            raise NotImplementedError("Subclass of BaseEnvironment does not implement 'duplicate()'")

        @abstractmethod
        def get_input_shape(self) -> (int, ...):
            """
            @return: Environment input shape that is required from the applied TF models
            """
            raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_input_shape()'")

        @abstractmethod
        def get_output_shape(self) -> (int, ...):
            """
            @return: Environment output shape that is required from the applied TF models
            """
            raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_output_shape()'")

