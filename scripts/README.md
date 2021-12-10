# Scripts

We provide a list of scripts to facilitate training, evaluation of models for WebNLG, TekGen.  
Here is the list of all scripts w/ description.

```
.
├── evaluate.sh                        # Evaluate model saved at end of training epoch (e.g. epoch=1, 2, ...)
├── evaluate.checkpoints_eval.sh       # Evaluate model checkpointed during training epoch (on fractional epochs)
|
├── generate.sh                        # Generate hypotheses for model saved at end of training epoch
├── generate.checkpoints_eval.sh       # Generate hypothese for model checkpointed during training epoch (on fractional epochs)
|
├── generate_rdf_references_test.sh    # Generate WebNLG graph references for test
├── generate_rdf_references_val.sh     # Generate WebNLG graph references for val
├── generate_references_test.sh        # Generate WebNLG text references for test
├── generate_references_val.sh         # Generate WebNLG text references for val
|
├── prepare.tekgen.g2t.scst.sh         # prepare CE model for TekGen SCST G2T training
├── prepare.tekgen.t2g.scst.sh         # prepare CE model for TekGen SCST T2G training
├── prepare.webnlg.g2t.scst.sh         # prepare CE model for WebNLG SCST G2T training
├── prepare.webnlg.t2g.scst.sh         # prepare CE model for WebNLG SCST T2G training
|
├── query_config.sh                    # Query values from .cfg files from CLI
|
├── train.sh                           # Main training script (called by wrapper scripts below)
├── train.tekgen.g2t.ce.sh             #   wrapper training script for TekGen G2T CE training 
├── train.tekgen.g2t.scst.sh           #   wrapper training script for TekGen G2T SCST training 
├── train.tekgen.t2g.ce.sh             #   wrapper training script for TekGen T2G CE training 
├── train.tekgen.t2g.scst.sh           #   wrapper training script for TekGen T2G SCST training 
├── train.webnlg.g2t.ce.sh             #   wrapper training script for TekGen G2T CE training 
├── train.webnlg.g2t.scst.sh           #   wrapper training script for TekGen G2T SCST training 
├── train.webnlg.t2g.ce.sh             #   wrapper training script for TekGen T2G CE training 
└── train.webnlg.t2g.scst.sh           #   wrapper training script for TekGen T2G SCST training 
```
