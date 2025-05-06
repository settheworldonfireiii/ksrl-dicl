#TODO:

1) MODIFY thin_data_new() function inside src/dicl/rl/NB_dx_tf_new.py so that it

1.1 does not delete the data after the container pruning has finished

2) MAKE thin_data_new() method of src/dicl/rl/NB_dx_tf_new.py return the indices of the best 300 datapoints WHEN THINNING NEW DATA ON SYNTHETIC DATA (real==False flag within the function)
this can be done either by
2.1 sampling 300 best points after initializing the container, without even going to the loop or inside ksdp/pruning_container.py
2.2 go inside ksdp and make it return 300 best points instead of n (currently 5) worst points as it is now

BE SURE TO ADJUST OTHER PARTS OF THE CODE THAT RELY ON THE CODE YOU MODIFIED, IF ANY

3) take best 300 datapoints in order to be fed to SAC (critic) training proces:
 either by using rb_llm and indices returned from thin_data_new(), 
OR  by returning the ACTUAL DATA from thin_data_new() (I suppose that one will be harder because you will have to compute next_observation by adding self.train_y_s, the difference, and extract actions from self.train_x_s)


4) BONUS: try implementing Bayesian thinning on ReplayBuffer so that we feed best 300 transitions to the LLM.

5) MEGABONUS : any ideas to speed things up while retaining quality?
