#

htk_root=/share/htk
openfst_root=/share/kaldi/tools/openfst
kaldi_root=/share/kaldi/src
vulcan_root=/share/vulcan

PATH=$htk_root/bin:$PATH
PATH=$openfst_root/bin:$PATH
PATH=$kaldi_root/bin:$PATH
PATH=$kaldi_root/fstbin/:$PATH
PATH=$kaldi_root/gmmbin/:$PATH
PATH=$kaldi_root/featbin/:$PATH
PATH=$kaldi_root/sgmmbin/:$PATH
PATH=$kaldi_root/sgmm2bin/:$PATH
PATH=$kaldi_root/fgmmbin/:$PATH
PATH=$kaldi_root/latbin/:$PATH
PATH=$kaldi_root/nnetbin/:$PATH
PATH=$vulcan_root/bin/:$PATH
PATH=$vulcan_root/HDecode++/:$PATH
PATH=$kaldi_root/lmbin/:$PATH
export PATH=$PATH

##
cmvn_dir=exp/cmvn

export dev_feat_setup="feat/dev.39.cmvn.ark"
export test_feat_setup="feat/test.39.cmvn.ark"
export train_feat_setup="feat/train.39.cmvn.ark"
