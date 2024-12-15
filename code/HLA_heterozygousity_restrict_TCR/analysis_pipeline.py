import pandas as pd
import numpy as np
from sonnia.processing import Processing
from sonnia.utils import gene_to_num_str
import righor
from tqdm import tqdm
import sys
from sonnia.sonia import Sonia
from sonnia.plotting import Plotter
import os
import multiprocessing

########## This code calculates the generation probability (Pgen) using IGOR and    
########## post-selection probability (Ppost) using SoNNia

## * Requires installation of IGOR and Sonia *

folder_with_files='path to raw repertoires dataset' ## contains Non-productive sequences

os.makedirs("igor_models", exist_ok=True)
os.makedirs("sonia_models", exist_ok=True)
def run_code(file):
    print(file)

    name=file.split('.')[0]
    if not os.path.exists("./igor_models/"+name):
        df=pd.read_csv(folder_with_files+file,sep='\t')
        use_cols=['rearrangement','amino_acid','templates','v_gene','j_gene','cdr3_rearrangement']
        df=df[use_cols]
        df['amino_acid'] = df['amino_acid'].replace('na', np.nan)
        df['v_gene']=df.v_gene.apply(lambda x: 'TRB'+gene_to_num_str(str(x),'V').upper())
        df['j_gene']=df.j_gene.apply(lambda x: 'TRB'+gene_to_num_str(str(x),'J').upper())

        #filter df
        pr=Processing(pgen_model='humanTRB')
        filtered=pr.filter_dataframe(df,read_col = 'templates', nt_seq_col = 'cdr3_rearrangement',apply_selection=False)

        productive=filtered.loc[filtered.selection]
        unproductive=filtered.loc[~filtered.selection_productive]
        unproductive_sequences=unproductive.sample(frac=1).rearrangement.values

        print('productive seqs: ',len(productive))
        print('unprodutive seqs: ',len(unproductive_sequences))

        if len(productive)<20000 or len(unproductive_sequences)<5000:
            print('error: not enough seqs here')
            sys.exit()
    
        # define righor model
        igor_model = righor.load_model("human", "trb")
        uniform_model = igor_model.uniform().copy()
    
        # define parameters for the alignment and the inference
        align_params = righor.AlignmentParameters()
        align_params.left_v_cutoff = 600
        align_params.min_score_v = -100000
        align_params.min_score_j = -100000
        align_params.max_error_d = 100
        infer_params = righor.InferenceParameters()
    
        print('Align seqs for IGoR')
        # read the file line by line and align each sequence
        np.random.shuffle(unproductive_sequences)
        alignments = igor_model.align_all_sequences(unproductive_sequences[:int(3e4)],align_params)
        print('Infer IGoR')
        # infer righor model
        models = []
        uniform_model = igor_model.uniform().copy()
    
        for rd in tqdm(range(10)):
            print(rd, end=" ")
            models.append(uniform_model.copy())
            uniform_model.infer(alignments, infer_params)
    
        models[-1].save_model('igor_models/'+name)
    
        fig = righor.plot_vdj(*( [models[-1]] +[igor_model]), plots_kws= [{'label':f'inferred'},{'label':f'default'}])
        fig.savefig('igor_models/'+name+'/model_pars.png')

    print('Define Sonia model')

    #define sonia model and infer
    sonia_model=Sonia(pgen_model='igor_models/'+name,data_seqs=productive[['amino_acid','v_gene','j_gene']].values,preprocess_seqs=False)
    sonia_model.add_generated_seqs(int(5e5))

    print('Infer Sonia model')

    sonia_model.infer_selection(epochs=80)
    sonia_model.save_model('sonia_models/'+name)

    pl=Plotter(sonia_model)
    pl.plot_model_learning(save_name='sonia_models/'+name+'/model_learning.png')
    pl.plot_vjl(save_name='sonia_models/'+name+'/vjl.png')
    pl.plot_logQ(save_name='sonia_models/'+name+'/logQ.png')

    print('Evaluate Pgen, PPost of generated sequences and compute model entropy')

    Q, pgens, pposts = sonia_model.evaluate_seqs(sonia_model.gen_seqs[:int(1e5)])
    sel = pgens > 0
    entropy_post=-np.mean(Q[sel] * np.log2(pposts[sel]))
    entropy_gen=-np.mean(np.log2(pgens[sel]))
    ## std errors: post +- 0.17, gen +- 0.02
    with open('sonia_models/'+name+'/entropies.csv','w') as f:
        f.writelines('entropy_gen,entropy_post\n')
        f.writelines(str(entropy_gen)[:7]+','+str(entropy_post)[:7]+'\n')
        
if __name__ == "__main__":  
    files = [x for x in os.listdir(folder_with_files) if x.endswith(".tsv")]
    with multiprocessing.Pool(processes=1) as pool:  # Adjust number of processes as needed
        pool.map(run_code, files)