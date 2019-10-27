/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */ 
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*      Machine Intelligence Laboratory                        */
/*      Department of Engineering                              */
/*      University of Cambridge                                */
/*      http://mi.eng.cam.ac.uk/                               */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright:                                          */
/*         2000-2003  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: HDecode.c  HTK Large Vocabulary Decoder       */
/* ----------------------------------------------------------- */

char *hdecode_version = "!HVER!HDecode:   3.4.1 [GE 12/03/09]";
char *hdecode_sccs_id = "$Id: HDecode.c,v 1.1.1.1 2006/10/11 09:54:55 jal58 Exp $";

/* this is just the tool that handles command line arguments and
   stuff, all the real magic is in HLVNet and HLVRec */


#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HLabel.h"
#include "HAudio.h"
#include "HParm.h"
#include "HDict.h"
#include "HModel.h"
#include "HUtil.h"
#include "HTrain.h"
#include "HAdapt.h"
#include "HNet.h"       /* for Lattice */
#include "HLat.h"       /* for Lattice */

#include "config.h"

#include "HLVNet.h"
#include "HLVRec.h"
#include "HLVLM.h"

#include <time.h>
#include <string>
using std::string;

bool
GetSenoneIndexMapping(const HMMSet& hset, 
                      const ContextDependency& tree,
                      const PhoneSet& phone_set,
                      vector<int32>& senone_map);
bool
InitializeExtraDataMembers(DecoderInst* dec);
bool
InitializeGMM(DecoderInst* dec,
              const string& gmm_mdl,
              const string& gmm_tree);
bool
InitializeMLLR(DecoderInst* dec,
               const string& regtree,
               const string& xform,
               const string& utt2spk);
bool
InitializeSGMM(DecoderInst* dec,
               const string& sgmm_mdl,
               const string& sgmm_tree,
               const string& sgmm_gelect,
               const string& sgmm_spk_vecs,
               const string& utt2spk);
bool
InitializeSGMM2(DecoderInst* dec,
                const string& sgmm2_mdl,
                const string& sgmm2_tree,
                const string& sgmm2_gelect,
                const string& sgmm2_spk_vecs,
                const string& utt2spk);
bool
InitializeAcousticFeature(DecoderInst* dec,
                          const string& feature);
bool
InitializeSenoneScoreTable(DecoderInst* dec,
                           const string& table,
                           const string& tree);
bool 
Prepare(DecoderInst* dec);
bool 
Done(DecoderInst* dec);
bool 
Next(DecoderInst* dec);
string 
UtteranceId(DecoderInst* dec);

vector<size_t>
HourMinSec(const float& sec);

static char* mlf_fn = NULL;
static char* mmf_fn = NULL;
static char* utt_fn = NULL;

/* -------------------------- Trace Flags & Vars ------------------------ */

#define T_TOP 00001		/* Basic progress reporting */
#define T_OBS 00002		/* Print Observation */
#define T_ADP 00004		/* Adaptation */
#define T_MEM 00010		/* Memory usage, start and finish */

static int trace = 1;

/* -------------------------- Global Variables etc ---------------------- */


static char *langfn;		/* LM filename from commandline */
static char *dictfn;		/* dict filename from commandline */
static char *hmmListfn;		/* model list filename from commandline */
static char *hmmDir = NULL;     /* directory to look for HMM def files */
static char *hmmExt = NULL;     /* HMM def file extension */

static FileFormat ofmt = UNDEFF;	/* Label output file format */
static char *labDir = NULL;	/* output label file directory */
static char *labExt = "rec";	/* output label file extension */
static char *labForm = NULL;	/* output label format */

static Boolean latRescore = FALSE; /* read lattice for each utterance and rescore? */
static char *latInDir = NULL;   /* lattice input directory */
static char *latInExt = "lat";  /* latttice input extension */
static char *latFileMask = NULL; /* mask for reading lattice */

static Boolean latGen = FALSE;  /* output lattice? */
static char *latOutDir = NULL;  /* lattice output directory */
static char *latOutExt = "lat"; /* latttice output extension */
static char *latOutForm = NULL;  /* lattice output format */

static FileFormat dataForm = UNDEFF; /* data input file format */

static Vocab vocab;		/* wordlist or dictionary */
static HMMSet hset;		/* HMM set */
static FSLM *lm;                /* language model */
static LexNet *net;             /* Lexicon network of all required words/prons */

static char *startWord = "<s>"; /* word used at start of network */
static LabId startLab;          /*   corresponding LabId */
static char *endWord = "</s>";  /* word used at end of network */
static LabId endLab;            /*   corresponding LabId */

static char *spModel = "sp";    /* model used as word end Short Pause */
static LabId spLab;             /*   corresponding LabId */
static char *silModel = "sil";  /* model used as word end Silence */
static LabId silLab;            /*   corresponding LabId */

static Boolean silDict = FALSE; /* does dict contain -/sp/sil variants with probs */

static LogFloat insPen = 0.0;   /* word insertion penalty */

static float acScale = 1.0;     /* acoustic scaling factor */
static float pronScale = 1.0;   /* pronunciation scaling factor */
static float lmScale = 1.0;     /* LM scaling factor */

static int maxModel = 0;        /* max model pruning */
static LogFloat beamWidth = - LZERO;     /* pruning global beam width */
static LogFloat weBeamWidth = - LZERO;   /* pruning wordend beam width */
static LogFloat zsBeamWidth = - LZERO;   /* pruning z-s beam width */
static LogFloat relBeamWidth = - LZERO;  /* pruning relative beam width */
static LogFloat latPruneBeam = - LZERO;  /* lattice pruning beam width */
static LogFloat latPruneAPS = 0;;        /* lattice pruning arcs per sec limit */

static LogFloat fastlmlaBeam = - LZERO;  /* do fast LM la outside this beam */

static int nTok = 32;           /* number of different LMStates per HMM state */
static Boolean useHModel = FALSE; /* use standard HModel OutP functions */
static int outpBlocksize = 1;   /* number of frames for which outP is calculated in one go */
static Observation *obs;        /* array of Observations */

/* transforms/adaptatin */
/* information about transforms */
static XFInfo xfInfo;


/* info for comparing scores from alignment of 1-best with search */
static char *bestAlignMLF;      /* MLF with 1-best alignment */

/* -------------------------- Heaps ------------------------------------- */

static MemHeap modelHeap;
static MemHeap netHeap;
static MemHeap lmHeap;
static MemHeap inputBufHeap;
static MemHeap transHeap;
static MemHeap regHeap;

/* -------------------------- Prototypes -------------------------------- */
void SetConfParms (void);
void ReportUsage (void);
DecoderInst *Initialise (void);
void DoRecognition (DecoderInst *dec, char *fn);
Boolean UpdateSpkrModels (char *fn);

/* ---------------- Configuration Parameters ---------------------------- */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;		/* total num params */


/* ---------------- Debug support  ------------------------------------- */

#if 0
FILE *debug_stdout = stdout;
FILE *debug_stderr = stderr;
#endif

/* ---------------- Process Command Line ------------------------- */

/* SetConfParms: set conf parms relevant to this tool */
void
SetConfParms (void)
{
   int i;
   double f;
   Boolean b;
   char buf[MAXSTRLEN];

   nParm = GetConfig ("HDECODE", TRUE, cParm, MAXGLOBS);
   if (nParm > 0) {
      if (GetConfInt (cParm, nParm, "TRACE", &i))
	 trace = i;
      if (GetConfStr (cParm, nParm, "STARTWORD", buf))
         startWord = CopyString (&gstack, buf);
      if (GetConfStr (cParm, nParm, "ENDWORD", buf))
         endWord = CopyString (&gstack, buf);
      if (GetConfFlt (cParm, nParm, "LATPRUNEBEAM", &f))
         latPruneBeam  = f;
      if (GetConfFlt (cParm, nParm, "FASTLMLABEAM", &f))
         fastlmlaBeam  = f;
      if (GetConfFlt (cParm, nParm, "LATPRUNEAPS", &f))
         latPruneAPS  = f;
      if (GetConfStr (cParm, nParm, "BESTALIGNMLF", buf))
         bestAlignMLF = CopyString (&gstack, buf);
      if (GetConfBool (cParm, nParm, "USEHMODEL",&b)) useHModel = b;
      if (GetConfStr(cParm,nParm,"LATFILEMASK",buf)) {
         latFileMask = CopyString(&gstack, buf);
      }
   }
}

int
main (int argc, char *argv[])
{
   char *s, *datafn;
   DecoderInst *dec;

   #ifdef HDECODE_MOD
   const char *usage =
        "Generate recognized ( results / lattices ) using Viterbi search ( with phone alignment )\n"
        "Usage:  Hybrid.HDecode.mod [options]\n";
   const char *program = "Hybrid.HDecode.mod";
   #else
   const char *usage =
        "Generate recognized ( results / lattices ) using Viterbi search\n"
        "Usage:  Hybrid.HDecode [options]\n";
   const char *program = "Hybrid.HDecode";
   #endif

   try {
       kaldi::ParseOptions po(usage);
       acScale = 0.1;
       po.Register("am-weight", &acScale,
                   "[parameter] scaling factor for acoustic log-likelihoods");
       lmScale = 1.0;
       po.Register("lm-weight", &lmScale,
                   "[parameter] scaling factor for n-gram probabilities");
       string arpa_lm = "";
       po.Register("arpa-lm", &arpa_lm,
                   "[file] n-gram language model in ARPA format");
       string lex = "";
       po.Register("lex", &lex,
                   "[file] lexicon");
       string htk_mmf = "";
       po.Register("htk-mmf", &htk_mmf,
                   "[file] acoustic model in HTK format");
       string htk_tiedlist = "";
       po.Register("htk-tiedlist", &htk_tiedlist,
                   "[file] tiedlist in HTK format");
       string gmm_mdl = "";
       po.Register("gmm-mdl", &gmm_mdl,
                   "[file] acoustic model (GMM) in Kaldi format");
       string sgmm_mdl = "";
       po.Register("sgmm-mdl", &sgmm_mdl,
                   "[file] acoustic model (Subspace GMM) in Kaldi format");
       string sgmm2_mdl = "";
       po.Register("sgmm2-mdl", &sgmm2_mdl,
                   "[file] acoustic model (Subspace GMM ver 2.0) in Kaldi format");
       string gmm_tree = "";
       po.Register("gmm-tree", &gmm_tree,
                   "[file] senone tree for GMM acoustic model in Kaldi format");
       string sgmm_tree = "";
       po.Register("sgmm-tree", &sgmm_tree,
                   "[file] senone tree for Subspace GMM acoustic model in Kaldi format");
       string sgmm2_tree = "";
       po.Register("sgmm2-tree", &sgmm2_tree,
                   "[file] senone tree for Subspace GMM 2.0 acoustic model in Kaldi format");
       string score_tree = "";
       po.Register("score-tree", &score_tree,
                   "[file] senone tree for senone score table in Kaldi format");
       string mllr_tree = "";
       po.Register("mllr-tree", &mllr_tree,
                   "[file] MLLR regression tree for GMM acoustic model adaptation in Kaldi format");
       string mllr_xform = "";
       po.Register("mllr-xform", &mllr_xform,
                   "[file] MLLR transform set for GMM acoustic model adaptation in Kaldi format");
       string phonelist = "";
       po.Register("phonelist", &phonelist,
                   "[file] phoneme <-> integer list in Kaldi format");
       string feature = "";
       po.Register("feature", &feature,
                   "[rspecifier] acoustic feature archive in Kaldi format");
       string senone_score = "";
       po.Register("senone-score", &senone_score, 
                   "[rspecifier] senone score table in Kaldi format");
       string mlf = "";
       po.Register("mlf", &mlf,
                   "[file] recognition results in HTK format");
       string rec = "";
       po.Register("rec", &rec,
                   "[to be done] recognition results in Kaldi format");
       string lat_dir = "";
       po.Register("lat-dir", &lat_dir,
                   "[path] directory for storing lattices in HTK format");
       float gmm_weight = 0.0;
       po.Register("gmm-weight", &gmm_weight,
                   "[parameter] scaling factor of senone score generated by Kaldi GMM acoustic models");
       float sgmm_weight = 0.0;
       po.Register("sgmm-weight", &sgmm_weight,
                   "[parameter] scaling factor of senone score generated by Kaldi Subspace GMM acoustic models");
       float sgmm2_weight = 0.0;
       po.Register("sgmm2-weight", &sgmm2_weight,
                   "[parameter] scaling factor of senone score generated by Kaldi Subspace GMM 2.0 acoustic models");
       float score_weight = 0.0;
       po.Register("score-weight", &score_weight,
                   "[parameter] scaling factor of senone score gived by the provided table");
       float transition_scale = 1.0;
       po.Register("transition-scale", &transition_scale,
                   "[parameter] scaling factor of senone transition probabilities in log domain");
       float loop_scale = 1.0;
       po.Register("loop-scale", &loop_scale,
                   "[parameter] scaling factor of senone loop probabilities in log domain");
       BaseFloat log_prune = 5.0;
       po.Register("log-prune", &log_prune,
                   "[parameter] pruning beam used to reduce number of exp() evaluations");
       string lab_form = "M";
       po.Register("lab-form", &lab_form,
                   "[parameter] output label format in form of HTK setting (NCSTWMX)");
       string lat_form = "tvaldm";
       po.Register("lat-form", &lat_form,
                   "[parameter] output lattice format in form of HTK setting (ABtvaldmnr)");
       insPen = 0.0;
       po.Register("ins-penalty", &insPen,
                   "[parameter] word insertion penalty during recognition");
       float beam = 13.0;
       po.Register("beam", &beam,
                   "[parameter] beamwidth during recognition");
       trace = 1;
       po.Register("trace", &trace,
                   "[parameter] trace level");
       int samp_rate = 100;
       po.Register("samp-rate", &samp_rate,
                   "[parameter] sampling rate in terms of total number in seconds eg. 1 s / 10 ms = 100");
       string utt2spk = "";
       po.Register("utt2spk", &utt2spk,
                   "[rspecifier]utterance to speaker map");
       string sgmm_spk_vecs = "";
       po.Register("sgmm-spk-vecs", &sgmm_spk_vecs,
                   "[rspecifier] speaker vectors for Subspace GMM model");
       string sgmm2_spk_vecs = "";
       po.Register("sgmm2-spk-vecs", &sgmm2_spk_vecs,
                   "[rspecifier] speaker vectors for Subspace GMM 2.0 model");
       string sgmm_gselect = "";
       po.Register("sgmm-gselect", &sgmm_gselect, 
                   "[rspecifier] pre-selected Gaussian indices for Subspace GMM likelihood computation");
       string sgmm2_gselect = "";
       po.Register("sgmm2-gselect", &sgmm2_gselect,
                   "[rspecifier] pre-selected Gaussian indices for Subspace GMM 2.0 likelihood computation");
       po.Read(argc, argv);
       if(argc < 2) {
           po.PrintUsage();
           return 1;
       }
       { // checking parameter specification is good
           // checking lexicon
           if(lex == "") { fprintf(LOG_STREAM, "ERROR: --lexicon=%s is not speficied\n", lex.c_str()); exit(1); }
           // checking language model
           if(arpa_lm == "") { fprintf(LOG_STREAM, "ERROR: --arpa-lm=%s is not speficied\n", arpa_lm.c_str()); exit(1); }
           // checking htk acoustic model
           if((htk_mmf == "")||(htk_tiedlist == "")) { fprintf(LOG_STREAM, "ERROR: htk acoustic models must be specified\n"); }
           // checking kaldi GMM acoustic model
           bool gmm_am = false;
           if((gmm_mdl != "")&&(gmm_tree != "")&&(phonelist != "")) { gmm_am = true; }
           // checking kaldi Subspace GMM acoustic model
           bool sgmm_am = false;
           if((sgmm_mdl != "")&&(sgmm_tree != "")&&(sgmm_gselect != "")&&(phonelist != "")) { sgmm_am = true; }
           // chekcing kaldi Subspace GMM 2.0 acoustic model
           bool sgmm2_am = false;
           if((sgmm2_mdl != "")&&(sgmm2_tree != "")&&(sgmm2_gselect != "")&&(phonelist != "")) { sgmm2_am = true; }
       }
       // htk structure initialization
       if (InitShell (argc, argv, hdecode_version, hdecode_sccs_id) < SUCCESS) {
           HError (4000, "HDecode: InitShell failed");
       }
       InitMem ();
       InitMath ();
       InitSigP ();
       InitWave ();
       InitLabel ();
       InitAudio ();
       InitModel ();
       if (InitParm () < SUCCESS) { HError (4000, "HDecode: InitParm failed"); }
       InitUtil ();
       InitDict ();
       InitLVNet ();
       InitLVLM ();
       InitLVRec ();
       InitAdapt (&xfInfo);
       InitLat ();
       SetConfParms ();
       CreateHeap(&modelHeap, "Model heap",  MSTAK, 1, 0.0, 100000, 800000 );
       CreateHMMSet(&hset,&modelHeap,TRUE);
       // decoder component initialization
       if(mlf != "") {
           mlf_fn = (char*) malloc((mlf.length()+1) * sizeof(char));
           strcpy(mlf_fn, mlf.c_str());
           if (SaveToMasterfile (mlf_fn) < SUCCESS) {
               HError (4014, "HDecode: Cannot write to MLF");
           }
       }
       if(rec != "") {
       }
       if(lat_dir != "") {
           latGen = true;
           latOutDir = (char*) malloc((lat_dir.length()+1) * sizeof(char));
           strcpy(latOutDir, lat_dir.c_str());
       }
       if(lab_form != "") {
           labForm = (char*) malloc((lab_form.length()+1) * sizeof(char));
           strcpy(labForm, lab_form.c_str());
       }
       if(lat_form != "") {
           latOutForm = (char*) malloc((lat_form.length()+1) * sizeof(char));
           strcpy(latOutForm, lat_form.c_str());
       }
       if (insPen > 0.0) {
           HError (-1, "HDecode: positive word insertion penalty???");
       }
       beamWidth = beam;
       if (latPruneBeam == -LZERO) {
           latPruneBeam = beamWidth;
       }
       relBeamWidth = beamWidth;
       if(arpa_lm != "") {
           langfn = (char*) malloc((arpa_lm.length()+1) * sizeof(char));
           strcpy(langfn, arpa_lm.c_str());
       }
       if(htk_mmf != "") {
           mmf_fn = (char*) malloc((htk_mmf.length()+1) * sizeof(char));
           strcpy(mmf_fn, htk_mmf.c_str());
           AddMMF (&hset, mmf_fn);
       }
       if(lex != "") {
           dictfn = (char*) malloc((lex.length()+1) * sizeof(char));
           strcpy(dictfn, lex.c_str());
       }
       if(htk_tiedlist != "") {
           hmmListfn = (char*) malloc((htk_tiedlist.length()+1) * sizeof(char));
           strcpy(hmmListfn, htk_tiedlist.c_str());
       }

       dec = Initialise ();

       // initialization of extra members in decoder
       InitializeExtraDataMembers(dec);
       dec->gmm_weight = gmm_weight;
       dec->sgmm_weight = sgmm_weight;
       dec->sgmm2_weight = sgmm2_weight;
       dec->score_weight = score_weight;
       dec->transition_scale = transition_scale;
       dec->loop_scale = loop_scale;
       dec->log_prune = log_prune;
       dec->samp_rate = samp_rate;
       if(phonelist != "") {
           dec->phone_set_ptr = new PhoneSet;
           dec->phone_set_ptr->Load(phonelist);
       }
       if(utt2spk != "") {
       }
       if((gmm_mdl != "")&&(gmm_tree != "")) {
           InitializeGMM(dec, gmm_mdl, gmm_tree);
           if((mllr_tree != "")&&(mllr_xform != "")&&(utt2spk != "")) {
               InitializeMLLR(dec, mllr_tree, mllr_xform, utt2spk);
           }
       }
       if((sgmm_mdl != "")&&(sgmm_tree != "")&&(sgmm_gselect != "")) {
           if((sgmm_spk_vecs != "")&&(utt2spk != "")) {
               InitializeSGMM(dec, sgmm_mdl, sgmm_tree,
                              sgmm_gselect, sgmm_spk_vecs, utt2spk);
           }
           else {
               InitializeSGMM(dec, sgmm_mdl, sgmm_tree,
                              sgmm_gselect, sgmm_spk_vecs, utt2spk);
           }
       }
       if((sgmm2_mdl != "")&&(sgmm2_tree != "")&&(sgmm2_gselect != "")) {
           if((sgmm2_spk_vecs != "")&&(utt2spk != "")) {
               InitializeSGMM2(dec, sgmm2_mdl, sgmm2_tree,
                               sgmm2_gselect, sgmm2_spk_vecs, utt2spk);
           }
           else {
               InitializeSGMM2(dec, sgmm2_mdl, sgmm2_tree,
                               sgmm2_gselect, sgmm2_spk_vecs, utt2spk);
           }
       }
       if(feature != "") {
           InitializeAcousticFeature(dec, feature);
       }
       if((senone_score != "")&&(score_tree != "")) {
           InitializeSenoneScoreTable(dec, senone_score, score_tree);
       }
       fprintf(LOG_STREAM, "\n");
       while(Done(dec) == false) {
           if(utt_fn != NULL) { free(utt_fn); }
           string fid = UtteranceId(dec);
           utt_fn = (char*) malloc((fid.length()+1)*sizeof(char));
           strcpy(utt_fn, fid.c_str());
           Prepare(dec);
           DoRecognition(dec, utt_fn);
           Next(dec);
           free(utt_fn);
           utt_fn = NULL;
       }
       // Clear(dec)
       vector<size_t> uttTime = HourMinSec(dec->uttAll);
       fprintf (LOG_STREAM, "%s : total utterance length = [ %zu ] hours [ %zu ] mins [ %zu ] secs\n",
                program, uttTime[0], uttTime[1], uttTime[2]);
       vector<size_t> cpuTime = HourMinSec(dec->cpuAll);
       fprintf (LOG_STREAM, "%s : total decoding time = [ %zu ] hours [ %zu ] mins [ %zu ] secs\n",
                program, cpuTime[0], cpuTime[1], cpuTime[2]);
       if(dec->latAll > 0.0) {
           vector<size_t> latTime = HourMinSec(dec->latAll);
           fprintf (LOG_STREAM, "%s : total lattice storage time = [ %zu ] hours [ %zu ] mins [ %zu ] secs\n",
                    program, latTime[0], latTime[1], latTime[2]);
       }
       float RTF = dec->cpuAll / dec->uttAll;
       fprintf (LOG_STREAM, "%s : total real time factor = [ %.2f ]\n", program, RTF);
       fprintf (LOG_STREAM, "\n");
   } catch(const std::exception &e) {
       std::cerr << e.what();
       return -1;
   }
   return 0;
}

bool
SplitTriphone(char* triphone, vector<string>& result) {
    string tri = string(triphone);
    size_t mpos = tri.find("-");
    if(mpos == string::npos) { result[0] = "<eps>"; mpos = 0; }
    else { result[0] = tri.substr(0, mpos); mpos++; }
    size_t ppos = tri.find("+");
    if(ppos == string::npos) { result[2] = "<eps>"; ppos = tri.length(); }
    else { result[2] = tri.substr(ppos+1); }
    result[1] = tri.substr(mpos, ppos-mpos);
    return true;
}
bool
GetSenoneIndexMapping(const HMMSet& hset,
                      const ContextDependency& tree,
                      const PhoneSet& phone_set,
                      vector<int32>& senone_map) {
    int32 context_width = tree.ContextWidth();
    vector<int32> phone_seq(context_width, 0);
    vector<string> phone_str_seq(context_width, "");
    int32 pdf_class = 0;
    int32 pdf_id = 0;
    bool success = false;
    for (int h = 0; h < MACHASHSIZE; h++) {
        for (MLink m = hset.mtab[h]; m != NULL; m = m->next) {
            if(m->type == 'h') {
                if(context_width == 1) {
                    string middle = string(m->id->name);
                    if(middle == "sp") { continue; }
                    phone_seq[0] = phone_set.index(middle);
                }
                if(context_width == 3) {
                    SplitTriphone(m->id->name, phone_str_seq);
                    if(phone_str_seq[1] == "sp") { continue; }
                    for(size_t n = 0; n < context_width; n++) {
                        phone_seq[n] = phone_set.index(phone_str_seq[n]);
                    }
                }
                HLink hl = (HLink) m->structure;
                for(short i = 2; i < hl->numStates; i++) {
                    StateInfo* info = (hl->svec+i)->info;
                    int sIdx = info->sIdx;
                    if(senone_map.size() <= sIdx) {
                        senone_map.resize(sIdx+1, -1);
                    }
                    if(senone_map[sIdx] != -1) {
                        continue;
                    }
                    pdf_class = i-2;
                    success = tree.Compute(phone_seq, pdf_class, &pdf_id);
                    if(success == true) {
                        senone_map[sIdx] = pdf_id;
                    }
/*
                    fprintf(stdout, "DEBUG: m->id->name = [ %s ] ", m->id->name);
                    fprintf(stdout, "phone_seq = ");
                    for(size_t n = 0; n < phone_seq.size(); n++) {
                        fprintf(stdout, "[ %d ] ", phone_seq[n]);
                    }
                    fprintf(stdout, "pdf_class = [ %d ]\n", pdf_class);
                    fprintf(stdout, "       success = [ %s ] ", success ? "true" : "false");
                    fprintf(stdout, "pdf_id = [ %d ] sIdx = [ %d ]\n", pdf_id, sIdx);
*/
                }
            }
        }
    }
    return true;
}

bool
InitializeExtraDataMembers(DecoderInst* dec) {
    dec->phone_set_ptr = NULL;
    dec->trans_gmm_ptr = NULL;
    dec->am_gmm_ptr = NULL;
    dec->tree_gmm_ptr = NULL;
    dec->senone_map_gmm_ptr = NULL;
    dec->mllr_regtree = NULL;
    dec->mllr_reader = NULL;
    dec->trans_sgmm_ptr = NULL;
    dec->am_sgmm_ptr = NULL;
    dec->tree_sgmm_ptr = NULL;
    dec->senone_map_sgmm_ptr = NULL;
    dec->spkvecs_reader_sgmm_ptr = NULL;
    dec->spk_vars_sgmm_ptr = NULL;
    dec->gselect_reader_sgmm_ptr = NULL;
    dec->trans_sgmm2_ptr = NULL;
    dec->am_sgmm2_ptr = NULL;
    dec->tree_sgmm2_ptr = NULL;
    dec->senone_map_sgmm2_ptr = NULL;
    dec->spkvecs_reader_sgmm2_ptr = NULL;
    dec->spk_vars_sgmm2_ptr = NULL;
    dec->gselect_reader_sgmm2_ptr = NULL;
    dec->score_interface_gmm_ptr = NULL;
    dec->score_interface_gmm_mllr_ptr = NULL;
    dec->score_interface_sgmm_ptr = NULL;
    dec->score_interface_sgmm2_ptr = NULL;
    dec->tree_score_ptr = NULL;
    dec->senone_map_score_ptr = NULL;
    dec->acoustic_feature = NULL;
    dec->senone_score_table = NULL;
    dec->numUtts = 0;
    dec->uttSec = 0.0;
    dec->cpuSec = 0.0;
    dec->latSec = 0.0;
    dec->uttAll = 0.0;
    dec->cpuAll = 0.0;
    dec->latAll = 0.0;
    return true;
}

bool
InitializeGMM(DecoderInst* dec,
              const string& gmm_mdl,
              const string& gmm_tree) {
    assert(dec->phone_set_ptr != NULL);
    dec->trans_gmm_ptr = new TransitionModel;
    dec->am_gmm_ptr = new AmDiagGmm;
    {
       bool binary;
       kaldi::Input ki(gmm_mdl, &binary);
       dec->trans_gmm_ptr->Read(ki.Stream(), binary);
       dec->am_gmm_ptr->Read(ki.Stream(), binary);
    }
    dec->tree_gmm_ptr = new ContextDependency;
    {
        bool binary;
        kaldi::Input ki(gmm_tree, &binary);
        dec->tree_gmm_ptr->Read(ki.Stream(), binary);
    }
    dec->senone_map_gmm_ptr = new vector<int32>;
    GetSenoneIndexMapping(hset, *(dec->tree_gmm_ptr), *(dec->phone_set_ptr), *(dec->senone_map_gmm_ptr));
    return true; 
}

bool
InitializeMLLR(DecoderInst* dec,
               const string& regtree,
               const string& xform,
               const string& utt2spk) {
    assert(dec->phone_set_ptr != NULL);
    assert(dec->am_gmm_ptr != NULL);
    dec->mllr_regtree = new RegressionTree;
    {
        bool binary;
        kaldi::Input in(regtree, &binary);
        dec->mllr_regtree->Read(in.Stream(), binary, *(dec->am_gmm_ptr));
    }
    dec->mllr_reader = new RandomAccessRegtreeMllrDiagGmmReaderMapped(xform, utt2spk);
    return true;
}
bool
InitializeSGMM(DecoderInst* dec,
               const string& sgmm_mdl,
               const string& sgmm_tree,
               const string& sgmm_gselect,
               const string& sgmm_spk_vecs,
               const string& utt2spk) {
    assert(dec->phone_set_ptr != NULL);
    dec->trans_sgmm_ptr = new TransitionModel;
    dec->am_sgmm_ptr = new AmSgmm;
    {
       bool binary;
       kaldi::Input ki(sgmm_mdl, &binary);
       dec->trans_sgmm_ptr->Read(ki.Stream(), binary);
       dec->am_sgmm_ptr->Read(ki.Stream(), binary);
    }
    dec->tree_sgmm_ptr = new ContextDependency;
    {
        bool binary;
        kaldi::Input ki(sgmm_tree, &binary);
        dec->tree_sgmm_ptr->Read(ki.Stream(), binary);
    }
    dec->senone_map_sgmm_ptr = new vector<int32>;
    GetSenoneIndexMapping(hset, *(dec->tree_sgmm_ptr), *(dec->phone_set_ptr), *(dec->senone_map_sgmm_ptr));
    if((sgmm_spk_vecs != "")&&(utt2spk != "")) {
        dec->spkvecs_reader_sgmm_ptr = new RandomAccessBaseFloatVectorReaderMapped(sgmm_spk_vecs, utt2spk);
    }
    dec->spk_vars_sgmm_ptr = new SgmmPerSpkDerivedVars;
    dec->gselect_reader_sgmm_ptr = new RandomAccessInt32VectorVectorReader(sgmm_gselect);
    return true;
}

bool
InitializeSGMM2(DecoderInst* dec,
                const string& sgmm2_mdl,
                const string& sgmm2_tree,
                const string& sgmm2_gselect,
                const string& sgmm2_spk_vecs,
                const string& utt2spk) {
    assert(dec->phone_set_ptr != NULL);
    dec->trans_sgmm2_ptr = new TransitionModel;
    dec->am_sgmm2_ptr = new AmSgmm2;
    {
       bool binary;
       kaldi::Input ki(sgmm2_mdl, &binary);
       dec->trans_sgmm2_ptr->Read(ki.Stream(), binary);
       dec->am_sgmm2_ptr->Read(ki.Stream(), binary);
    }
    dec->tree_sgmm2_ptr = new ContextDependency;
    {
        bool binary;
        kaldi::Input ki(sgmm2_tree, &binary);
        dec->tree_sgmm2_ptr->Read(ki.Stream(), binary);
    }
    dec->senone_map_sgmm2_ptr = new vector<int32>;
    GetSenoneIndexMapping(hset, *(dec->tree_sgmm2_ptr), *(dec->phone_set_ptr), *(dec->senone_map_sgmm2_ptr));
    if((sgmm2_spk_vecs != "")&&(utt2spk != "")) {
        dec->spkvecs_reader_sgmm2_ptr = new RandomAccessBaseFloatVectorReaderMapped(sgmm2_spk_vecs, utt2spk);
    }
    dec->spk_vars_sgmm2_ptr = new Sgmm2PerSpkDerivedVars;
    dec->gselect_reader_sgmm2_ptr = new RandomAccessInt32VectorVectorReader(sgmm2_gselect);
    return true;
}

bool
InitializeAcousticFeature(DecoderInst* dec,
                  const string& feature) {
    dec->acoustic_feature = new SequentialBaseFloatMatrixReader(feature);
    return true;
}

bool
InitializeSenoneScoreTable(DecoderInst* dec,
                           const string& table,
                           const string& tree) {
    assert(dec->phone_set_ptr != NULL);
    dec->tree_score_ptr = new ContextDependency;
    {  
        bool binary;
        kaldi::Input ki(tree, &binary);
        dec->tree_score_ptr->Read(ki.Stream(), binary);
    }   
    dec->senone_score_table = new SequentialBaseFloatMatrixReader(table);
    dec->senone_map_score_ptr = new vector<int32>;
    GetSenoneIndexMapping(hset, *(dec->tree_score_ptr), *(dec->phone_set_ptr), *(dec->senone_map_score_ptr));
    return true;
}

bool
Prepare(DecoderInst* dec) {
    if((dec->am_gmm_ptr != NULL)&&(dec->trans_gmm_ptr != NULL)&&(dec->acoustic_feature != NULL)) {
        if(dec->score_interface_gmm_ptr != NULL) { delete dec->score_interface_gmm_ptr; }
        if(dec->score_interface_gmm_mllr_ptr != NULL) { delete dec->score_interface_gmm_mllr_ptr; }
        const kaldi::Matrix<BaseFloat>& features = dec->acoustic_feature->Value();
//        dec->score_interface_gmm_ptr = 
//            new SenoneScoreAmDiagGmm(*(dec->am_gmm_ptr), *(dec->trans_gmm_ptr),
//                                     features, *(dec->senone_map_gmm_ptr), dec->log_prune);
        if((dec->mllr_regtree != NULL)&&(dec->mllr_reader != NULL)) {
//            if(dec->score_interface_gmm_mllr_ptr != NULL) { delete dec->score_interface_gmm_mllr_ptr; }
            const kaldi::Matrix<BaseFloat>& features = dec->acoustic_feature->Value();
            string fid = dec->acoustic_feature->Key();
            if(dec->mllr_reader->HasKey(fid)) {
                const RegtreeMllrDiagGmm& mllr_xform = dec->mllr_reader->Value(fid);
                dec->score_interface_gmm_mllr_ptr = 
                    new SenoneScoreAmDiagGmmRegtreeMllr(*(dec->am_gmm_ptr), *(dec->trans_gmm_ptr), 
                                                        features, mllr_xform, *(dec->mllr_regtree), 
                                                        *(dec->senone_map_gmm_ptr), 
                                                        dec->gmm_weight, dec->log_prune);
            }
        }
        else {
//            if(dec->score_interface_gmm_ptr != NULL) { delete dec->score_interface_gmm_ptr; }
            dec->score_interface_gmm_ptr = 
                new SenoneScoreAmDiagGmm(*(dec->am_gmm_ptr), *(dec->trans_gmm_ptr),
                                         features, *(dec->senone_map_gmm_ptr), dec->log_prune);
        }
    }
    if((dec->am_sgmm_ptr != NULL)&&(dec->trans_sgmm_ptr != NULL)&&(dec->acoustic_feature != NULL)&&(dec->gselect_reader_sgmm_ptr != NULL)) {
        if(dec->score_interface_sgmm_ptr != NULL) { delete dec->score_interface_sgmm_ptr; }
        string fid = UtteranceId(dec);
        assert(dec->gselect_reader_sgmm_ptr->HasKey(fid) == true);
        const kaldi::Matrix<BaseFloat>& features = dec->acoustic_feature->Value();
        const vector<vector<int32> >& gselect = dec->gselect_reader_sgmm_ptr->Value(fid);
        assert(gselect.size() == features.NumRows());
        if(dec->spkvecs_reader_sgmm_ptr != NULL) {
            assert(dec->spkvecs_reader_sgmm_ptr->HasKey(fid) == true);
            dec->spk_vars_sgmm_ptr->v_s = dec->spkvecs_reader_sgmm_ptr->Value(fid);
            dec->am_sgmm_ptr->ComputePerSpkDerivedVars(dec->spk_vars_sgmm_ptr);
        }
        SgmmGselectConfig sgmm_opts;
        dec->score_interface_sgmm_ptr = 
            new SenoneScoreAmSgmm(sgmm_opts, *(dec->am_sgmm_ptr), *(dec->spk_vars_sgmm_ptr),
                                  *(dec->trans_sgmm_ptr), features, gselect,
                                  *(dec->senone_map_sgmm_ptr), dec->log_prune);
    }
    if((dec->am_sgmm2_ptr != NULL)&&(dec->trans_sgmm2_ptr != NULL)&&(dec->acoustic_feature != NULL)&&(dec->gselect_reader_sgmm2_ptr != NULL)) {
        if(dec->score_interface_sgmm2_ptr != NULL) { delete dec->score_interface_sgmm2_ptr; }
        string fid = UtteranceId(dec);
        assert(dec->gselect_reader_sgmm2_ptr->HasKey(fid) == true);
        const kaldi::Matrix<BaseFloat>& features = dec->acoustic_feature->Value();
        const vector<vector<int32> >& gselect = dec->gselect_reader_sgmm2_ptr->Value(fid);
        assert(gselect.size() == features.NumRows());
        if(dec->spkvecs_reader_sgmm2_ptr != NULL) {
            assert(dec->spkvecs_reader_sgmm2_ptr->HasKey(fid) == true);
            dec->spk_vars_sgmm2_ptr->SetSpeakerVector(dec->spkvecs_reader_sgmm2_ptr->Value(fid));
            dec->am_sgmm2_ptr->ComputePerSpkDerivedVars(dec->spk_vars_sgmm2_ptr);
        }
        dec->score_interface_sgmm2_ptr =
            new SenoneScoreAmSgmm2(*(dec->am_sgmm2_ptr), *(dec->trans_sgmm2_ptr),
                                   features, gselect, *(dec->senone_map_sgmm2_ptr),
                                   dec->log_prune, dec->spk_vars_sgmm2_ptr);
    }
    return true;
}

bool 
Done(DecoderInst* dec) {
    if((dec->acoustic_feature == NULL)&&(dec->senone_score_table == NULL)) {
        return true;
    }
    if(dec->acoustic_feature != NULL) {
        return dec->acoustic_feature->Done();
    }
    return dec->senone_score_table->Done();
}

bool Next(DecoderInst* dec) {
    if(dec->acoustic_feature != NULL) {
        dec->acoustic_feature->Next();
    }
    if(dec->senone_score_table != NULL) {
        dec->senone_score_table->Next();
    }
    return true;
}

string UtteranceId(DecoderInst* dec) {
    string fid = "";
    if(dec->acoustic_feature != NULL) {
        fid = dec->acoustic_feature->Key();
    }
    if(dec->senone_score_table != NULL) {
        if((dec->acoustic_feature != NULL)&&(fid != dec->senone_score_table->Key())) {
            fprintf(LOG_STREAM, "ERROR: mismatch uttterance ID between acoustic feature [ %s ] and senone score table [ %s ]\n",
                                fid.c_str(), dec->senone_score_table->Key().c_str());
        }
        else {
            fid = dec->senone_score_table->Key();
        }
    }
    return fid;
}

vector<size_t>
HourMinSec(const float& sec) {
    size_t t = sec + 0.5;
    vector<size_t> result(3, 0);
    result[2] = t % 60;
    t -= result[2];
    t /= 60;
    result[1] = t % 60;
    t -= result[1];
    t /= 60;
    result[0] = t;
    return result;
}

DecoderInst *Initialise (void)
{
   int i;
   DecoderInst *dec;
   Boolean eSep;
   Boolean modAlign;

   /* init Heaps */
   CreateHeap (&netHeap, "Net heap", MSTAK, 1, 0,100000, 800000);
   CreateHeap (&lmHeap, "LM heap", MSTAK, 1, 0,1000000, 10000000);
   CreateHeap(&transHeap,"Transcription heap",MSTAK,1,0,8000,80000);

   /* Read dictionary */
   if (trace & T_TOP) {
      fprintf (LOG_STREAM, "Reading dictionary from %s\n", dictfn);
      fflush (stdout);
   }

   InitVocab (&vocab);
   if (ReadDict (dictfn, &vocab) < SUCCESS)
      HError (9999, "Initialise: ReadDict failed");

   /* Read accoustic models */
   if (trace & T_TOP) {
      fprintf (LOG_STREAM, "Reading acoustic models...");
      fflush (stdout);
   }
   if (MakeHMMSet (&hset, hmmListfn) < SUCCESS) 
      HError (4128, "Initialise: MakeHMMSet failed");
   if (LoadHMMSet (&hset, hmmDir, hmmExt) < SUCCESS) 
      HError (4128, "Initialise: LoadHMMSet failed");
   
   /* convert to INVDIAGC */
   ConvDiagC (&hset, TRUE);
   ConvLogWt (&hset);
   
   if (trace&T_TOP) {
      fprintf(LOG_STREAM, "Read %d physical / %d logical HMMs\n",
	     hset.numPhyHMM, hset.numLogHMM);  
      fflush (stdout);
   }

   /* process dictionary */
   startLab = GetLabId (startWord, FALSE);
   if (!startLab) 
      HError (9999, "HDecode: cannot find STARTWORD '%s'\n", startWord);
   endLab = GetLabId (endWord, FALSE);
   if (!endLab) 
      HError (9999, "HDecode: cannot find ENDWORD '%s'\n", endWord);

   spLab = GetLabId (spModel, FALSE);
   if (!spLab)
      HError (9999, "HDecode: cannot find label 'sp'");
   silLab = GetLabId (silModel, FALSE);
   if (!silLab)
      HError (9999, "HDecode: cannot find label 'sil'");


   if (silDict) {    /* dict contains -/sp/sil variants (with probs) */
      ConvertSilDict (&vocab, spLab, silLab, startLab, endLab);

      /* check for skip in sp model */
      { 
         LabId spLab;
         HLink spHMM;
         MLink spML;
         int N;

         spLab = GetLabId ("sp", FALSE);
         if (!spLab)
            HError (9999, "cannot find 'sp' model.");

         spML = FindMacroName (&hset, 'l', spLab);
         if (!spML)
            HError (9999, "cannot find model for sp");
         spHMM = (HLink) spML->structure;
         N = spHMM->numStates;

         if (spHMM->transP[1][N] > LSMALL)
            HError (9999, "HDecode: using -/sp/sil dictionary but sp contains tee transition!");
      }
   }
   else {       /* lvx-style dict (no sp/sil at wordend */
      MarkAllProns (&vocab);
   }
   

   if (!latRescore) {

      if (!langfn)
         HError (9999, "HDecode: no LM or lattice specified");

      /* mark all words  for inclusion in Net */
      MarkAllWords (&vocab);

      /* create network */
      net = CreateLexNet (&netHeap, &vocab, &hset, startWord, endWord, silDict);
      
      /* Read language model */
      if (trace & T_TOP) {
         fprintf (LOG_STREAM, "Reading language model from %s\n", langfn);
         fflush (stdout);
      }
      
      lm = CreateLM (&lmHeap, langfn, startWord, endWord, &vocab);
   }
   else {
      net = NULL;
      lm = NULL;
   }

   modAlign = FALSE;
   if (latOutForm) {
      if (strchr (latOutForm, 'd'))
         modAlign = TRUE;
      if (strchr (latOutForm, 'n'))
         HError (9999, "DoRecognition: likelihoods for model alignment not supported");
   }

   /* create Decoder instance */
   dec = CreateDecoderInst (&hset, lm, nTok, TRUE, useHModel, outpBlocksize,
                            bestAlignMLF ? TRUE : FALSE,
                            modAlign);
   
   /* create buffers for observations */
   SetStreamWidths (hset.pkind, hset.vecSize, hset.swidth, &eSep);

   obs = (Observation *) New (&gcheap, outpBlocksize * sizeof (Observation));
   for (i = 0; i < outpBlocksize; ++i)
      obs[i] = MakeObservation (&gcheap, hset.swidth, hset.pkind, 
                                (hset.hsKind == DISCRETEHS), eSep);

   CreateHeap (&inputBufHeap, "Input Buffer Heap", MSTAK, 1, 1.0, 80000, 800000);

   /* Initialise adaptation */

   /* sort out masks just in case using adaptation */
   if (xfInfo.inSpkrPat == NULL) xfInfo.inSpkrPat = xfInfo.outSpkrPat; 
   if (xfInfo.paSpkrPat == NULL) xfInfo.paSpkrPat = xfInfo.outSpkrPat; 

   if (xfInfo.useOutXForm) {
      CreateHeap(&regHeap,   "regClassStore",  MSTAK, 1, 0.5, 1000, 8000 );
      /* This initialises things - temporary hack - THINK!! */
      CreateAdaptXForm(&hset, "tmp");

      /* online adaptation not supported yet! */
   }


#ifdef LEGACY_CUHTK2_MLLR
   /* initialise adaptation */
   if (mllrTransDir) {
      CreateHeap(&regHeap,   "regClassStore",  MSTAK, 1, 0.5, 80000, 80000 );
      rt = (RegTransInfo *) New(&regHeap, sizeof(RegTransInfo));
      rt->nBlocks = 0;
      rt->classKind = DEF_REGCLASS;
      rt->adptSil = TRI_UNDEF;
      rt->nodeOccThresh = 0.0;

      /*# legacy CU-HTK adapt: create RegTree from INCORE/CLASS files
          and strore in ~r macro */
      LoadLegacyRegTree (&hset);
      
      InitialiseTransform(&hset, &regHeap, rt, FALSE);
   }
#endif

   return dec;
}


/**********  align best code  ****************************************/

/* linked list storing the info about the 1-best alignment read from BESTALIGNMLF 
   one bestInfo struct per model */
typedef struct _BestInfo BestInfo;
struct _BestInfo {
   int start;           /* frame numbers */
   int end;
   LexNode *ln;
   LLink ll;           /* get rid of this? currently start/end are redundant */
   BestInfo *next;
};


/* find the LN_MODEL lexnode following ln that has label lab
   step over LN_CON and LN_WORDEND nodes.
   return NULL if not found
*/
BestInfo *FindLexNetLab (MemHeap *heap, LexNode *ln, LLink ll, HTime frameDur)
{
   int i;
   LexNode *follLN;
   MLink m;
   BestInfo *info, *next;

   if (!ll->succ) {
      info = (BestInfo*) New (heap, sizeof (BestInfo));
      info->next = NULL;
      info->ll = NULL;
      info->ln = NULL;
      info->start = info->end = 0;
      return info;
   }
   
   for (i = 0; i < ln->nfoll; ++i) {
      follLN = ln->foll[i];
      if (follLN->type == LN_MODEL) {
         m = FindMacroStruct (&hset, 'h', follLN->data.hmm);
         if (m->id == ll->labid) {
            /*            fprintf (LOG_STREAM, "found  %8.0f %8.0f %8s  %p\n", ll->start, ll->end, ll->labid->name, follLN); */
            next = FindLexNetLab (heap, follLN, ll->succ, frameDur);
            if (next) {
               info = (BestInfo*) New (heap, sizeof (BestInfo));
               info->next = next;
               info->start = ll->start / (frameDur*1.0e7);
               info->end = ll->end / (frameDur*1.0e7);
               info->ll = ll;
               info->ln = follLN;
               return info;
            }
            /*            fprintf (LOG_STREAM, "damn got 0 back searching for %8s\n", ll->labid->name); */
         }
      }
      else {
         /*         fprintf (LOG_STREAM, "searching for %8s recursing\n", ll->labid->name); */
         next = FindLexNetLab (heap, follLN, ll, frameDur);
         if (next) {
            info = (BestInfo*) New (heap, sizeof (BestInfo));
            info->next = next;
            info->start = info->end = ll->start / (frameDur*1.0e7);
            info->ll = ll;
            info->ln = follLN;
            return info;
         }
         /*         fprintf (LOG_STREAM, "damn got 0 back from recursion\n"); */
      }
   }
   
   return NULL;
}

BestInfo *CreateBestInfo (MemHeap *heap, char *fn, HTime frameDur)
{
   char alignFN[MAXFNAMELEN];
   Transcription *bestTrans;
   LLink ll;
   LexNode *ln;
   MLink m;
   LabId lnLabId;
   BestInfo *bestAlignInfo;

   MakeFN (fn, "", "rec", alignFN);
   bestTrans = LOpen (&transHeap, alignFN, HTK);
      
   /* delete 'sp' or 'sil' before final 'sil' if it is there
      these are always inserted by HVite but not possible in HDecode's net structure*/
   if (bestTrans->head->tail->pred->pred->labid == spLab ||
       bestTrans->head->tail->pred->pred->labid == silLab) {
      LLink delLL;
      
      delLL = bestTrans->head->tail->pred->pred;
      /* add sp's frames (if any) to final sil */
      delLL->succ->start = delLL->pred->end;
      
      delLL->pred->succ = delLL->succ;
      delLL->succ->pred = delLL->pred;
   }
   
   ln = net->start;
   assert (ln->type == LN_MODEL);
   m = FindMacroStruct (&hset, 'h', ln->data.hmm);
   lnLabId = m->id;
   
   /* info for net start node */
   ll = bestTrans->head->head->succ;
#if 0
   fprintf (LOG_STREAM, "%8.0f %8.0f %8s   ln %p %8s\n", ll->start, ll->end, ll->labid->name, 
           ln, lnLabId->name);
#endif
   assert (ll->labid == lnLabId);
   bestAlignInfo = (BestInfo*) New (&transHeap, sizeof (BestInfo));
   bestAlignInfo->start = ll->start / (frameDur*1.0e7);
   bestAlignInfo->end = ll->end / (frameDur*1.0e7);
   bestAlignInfo->ll = ll;
   bestAlignInfo->ln = ln;
   
   
   /* info for all the following nodes */
   bestAlignInfo->next = FindLexNetLab (&transHeap, ln, ll->succ, frameDur);
   
   {
      BestInfo *b;
      for (b = bestAlignInfo; b->next; b = b->next)
         fprintf (LOG_STREAM, "%d %d %8s %p\n", b->start, b->end, b->ll->labid->name, b->ln);
   }

   return bestAlignInfo;
}

void PrintAlignBestInfo (DecoderInst *dec, BestInfo *b)
{
   LexNodeInst *inst;
   TokScore score;
   int l;
   LabId monoPhone;
   LogDouble phonePost;

   inst = b->ln->inst;
   score = inst ? inst->best : LZERO;

   if (b->ln->type == LN_MODEL) {
      monoPhone =(LabId) b->ln->data.hmm->hook;
      phonePost = dec->phonePost[(intptr_t) monoPhone->aux];
   } else
      phonePost = 999.99;

   l = dec->nLayers-1;
   while (dec->net->layerStart[l] > b->ln) {
      --l;
      assert (l >= 0);
   }
   
   fprintf (LOG_STREAM, "BESTALIGN frame %4d best %.3f alignbest %d -> %d ln %p layer %d score %.3f phonePost %.3f\n", 
           dec->frame, dec->bestScore, 
           b->start, b->end, b->ln, l, score, phonePost);
}

void AnalyseSearchSpace (DecoderInst *dec, BestInfo *bestInfo)
{
   BestInfo *b;
   LabId monoPhone;

   monoPhone =(LabId) dec->bestInst->node->data.hmm->hook;
   fprintf (LOG_STREAM, "frame %4d best %.3f phonePost %.3f\n", dec->frame, 
           dec->bestScore, dec->phonePost[(intptr_t) monoPhone->aux]);
 
   for (b = bestInfo; b; b = b->next) {
      if (b->start < dec->frame && b->end >= dec->frame) 
         break;
   }
   if (b) {
      PrintAlignBestInfo (dec, b);
      for (b = b->next; b && b->start == b->end && b->start == dec->frame; b = b->next) {
         PrintAlignBestInfo (dec, b);
      }
   }
   else {
      fprintf (LOG_STREAM, "BESTALIGN ERROR\n");
   }
}

/*****************  main recognition function  ************************/

void DoRecognition (DecoderInst *dec, char *fn)
{
   char buf1[MAXSTRLEN], buf2[MAXSTRLEN];
   ParmBuf parmBuf;
   BufferInfo pbInfo;
   int frameN, frameProc, i, bs;
   Transcription *trans;
   Lattice *lat;
   clock_t startClock, endClock;
   Observation *obsBlock[MAXBLOCKOBS];

   startClock = clock();

   pbInfo.srcSampRate = dec->samp_rate * 1000;
   pbInfo.tgtSampRate = dec->samp_rate * 1000;

   if (weBeamWidth > beamWidth)
      weBeamWidth = beamWidth;
   if (zsBeamWidth > beamWidth)
      zsBeamWidth = beamWidth;

   InitDecoderInst (dec, net, pbInfo.tgtSampRate, beamWidth, relBeamWidth,
                    weBeamWidth, zsBeamWidth, maxModel,
                    insPen, acScale, pronScale, lmScale, fastlmlaBeam);

   net->vocabFN = dictfn;
   dec->utterFN = fn;

   size_t numFrames = (dec->acoustic_feature != NULL)? dec->acoustic_feature->Value().NumRows() :
                      dec->senone_score_table->Value().NumRows() ;

   dec->numUtts++;
   dec->uttSec = dec->frameDur * numFrames;
   dec->uttAll += dec->uttSec;
   fprintf (LOG_STREAM, "File [ %05zu ] : %s , length = [ %.2f ] secs\n", dec->numUtts, fn, dec->uttSec);

   frameN = frameProc = 0;
   for(size_t i = 0; i < numFrames; i++) {
      ProcessFrame (dec, obsBlock, outpBlocksize, xfInfo.inXForm);
      ++frameProc;
      ++frameN;
   }

   assert (frameProc == frameN);
   
//   endClock = clock();
//   cpuSec = (endClock - startClock) / (double) CLOCKS_PER_SEC;
//   fprintf (LOG_STREAM, "CPU time %f  utterance length %f  RT factor %f\n",
//           cpuSec, frameN*dec->frameDur, cpuSec / (frameN*dec->frameDur));

   trans = TraceBack (&transHeap, dec);

   /* save 1-best transcription */
   /* the following is from HVite.c */
   if (trans) {
      char labfn[MAXSTRLEN];

      if (labForm != NULL)
         ReFormatTranscription (trans, pbInfo.tgtSampRate, FALSE, FALSE,
                                strchr(labForm,'X')!=NULL,
                                strchr(labForm,'N')!=NULL,strchr(labForm,'S')!=NULL,
                                strchr(labForm,'C')!=NULL,strchr(labForm,'T')!=NULL,
                                strchr(labForm,'W')!=NULL,strchr(labForm,'M')!=NULL);
      
      MakeFN (fn, labDir, labExt, labfn);

      if (LSave (labfn, trans, ofmt) < SUCCESS)
         HError(9999, "DoRecognition: Cannot save file %s", labfn);
/*
      if (trace & T_TOP)
         PrintTranscription (trans, "1-best hypothesis");

      Dispose (&transHeap, trans);
*/
   }

   endClock = clock();
   dec->cpuSec = (endClock - startClock) / (double) CLOCKS_PER_SEC;
   dec->cpuAll += dec->cpuSec;

   fprintf (LOG_STREAM, "Time : ");
   fprintf (LOG_STREAM, "decoding time = [ %.2f ] secs , ", dec->cpuSec);
   fprintf (LOG_STREAM, "real time factor = [ %.2f ]", dec->cpuSec / (frameN*dec->frameDur));

   if (latGen) {
      clock_t latStartClock = clock();
      lat = LatTraceBack (&transHeap, dec);

      /* prune lattice */
      if (lat && latPruneBeam < - LSMALL) {
         lat = LatPrune (&transHeap, lat, latPruneBeam, latPruneAPS);
      }

      /* the following is from HVite.c */
      if (lat) {
         char latfn[MAXSTRLEN];
         char *p;
         Boolean isPipe;
         FILE *file;
         LatFormat form;
         
         MakeFN (fn, latOutDir, latOutExt, latfn);
         file = FOpen (latfn, NetOFilter, &isPipe);
         if (!file) 
            HError (999, "DoRecognition: Could not open file %s for lattice output",latfn);
         if (!latOutForm)
            form = (HLAT_DEFAULT & ~HLAT_ALLIKE)|HLAT_PRLIKE;
         else {
            for (p = latOutForm, form=0; *p != 0; p++) {
               switch (*p) {
               case 'A': form|=HLAT_ALABS; break;
               case 'B': form|=HLAT_LBIN; break;
               case 't': form|=HLAT_TIMES; break;
               case 'v': form|=HLAT_PRON; break;
               case 'a': form|=HLAT_ACLIKE; break;
               case 'l': form|=HLAT_LMLIKE; break;
               case 'd': form|=HLAT_ALIGN; break;
               case 'm': form|=HLAT_ALDUR; break;
               case 'n': form|=HLAT_ALLIKE; 
                  HError (9999, "DoRecognition: likelihoods for model alignment not supported");
                  break;
               case 'r': form|=HLAT_PRLIKE; break;
               }
            }
         }
         lat->acscale = acScale;
         if (WriteLattice (lat, file, form) < SUCCESS)
            HError(9999, "DoRecognition: WriteLattice failed");
         
         FClose (file,isPipe);
         Dispose (&transHeap, lat);
      }
      clock_t latEndClock = clock();
      dec->latSec = (latEndClock - latStartClock) / (double) CLOCKS_PER_SEC;
      dec->latAll += dec->latSec;
      fprintf (LOG_STREAM, " , lattice storage time = [ %.2f ] secs", dec->latSec);
   }
   fprintf (LOG_STREAM, "\n");

   if (trans) {
      if (trace & T_TOP) {
         PrintTranscription (trans, "1-best hypothesis");
      }
      Dispose (&transHeap, trans);
   }

   fprintf (LOG_STREAM, "\n");
/*
#ifdef COLLECT_STATS
   fprintf (LOG_STREAM, "Stats: nTokSet %lu\n", dec->stats.nTokSet);
   fprintf (LOG_STREAM, "Stats: TokPerSet %f\n", dec->stats.sumTokPerTS / (double) dec->stats.nTokSet);
   fprintf (LOG_STREAM, "Stats: activePerFrame %f\n", dec->stats.nActive / (double) dec->stats.nFrames);
   fprintf (LOG_STREAM, "Stats: activateNodePerFrame %f\n", dec->stats.nActivate / (double) dec->stats.nFrames);
   fprintf (LOG_STREAM, "Stats: deActivateNodePerFrame %f\n\n", 
           dec->stats.nDeActivate / (double) dec->stats.nFrames);
#if 0
   fprintf (LOG_STREAM, "Stats: LMlaCacheHits %ld\n", dec->stats.nLMlaCacheHit);
   fprintf (LOG_STREAM, "Stats: LMlaCacheMiss %ld\n", dec->stats.nLMlaCacheMiss);
#endif
#ifdef COLLECT_STATS_ACTIVATION
   {
      int i;
      for (i = 0; i <= STATS_MAXT; ++i)
         fprintf (LOG_STREAM, "T %d Dead %lu Live %lu\n", i, dec->stats.lnDeadT[i], dec->stats.lnLiveT[i]);
   }
#endif
#endif
*/

   if (trace & T_MEM) {
      fprintf (LOG_STREAM, "memory stats at end of recognition\n");
      PrintAllHeapStats ();
   }

   ResetHeap (&inputBufHeap);
   ResetHeap (&transHeap);
   CleanDecoderInst (dec);
}

#ifdef LEGACY_CUHTK2_MLLR
void ResetFVTrans (HMMSet *hset, BlockMatrix transMat)
{
   HError (9999, "HDecode: switching speakers/transforms not supprted, yet");
}

void LoadFVTrans (char *fn, BlockMatrix *transMat)
{
   Source src;
   int blockSize, bs, i;
   short nblocks;
   char buf[MAXSTRLEN];
   Boolean binary = FALSE;
   
   InitSource (fn, &src, NoFilter);

   /* #### the file input should use HModel's/HAdapt's scanner  */
   ReadUntilLine (&src, "~k \"globalSemi\"");
   if (!ReadString (&src, buf) || strcmp (buf, "<SEMICOVAR>"))
      HError (9999, "LoadSemiTrans: expected <SEMICOVAR> tag in file '%s'");
   ReadShort (&src, &nblocks, 1, binary);

   ReadInt (&src, &blockSize, 1, binary);
   for(i = 2; i <= nblocks; i++) {
      ReadInt (&src, &bs, 1, binary);
      if (bs != blockSize)
         HError (9999, "LoadSemiTrans: BlockMats with different size blocks not supported");
   }
   if(!*transMat)
      *transMat = CreateBlockMat (&gcheap, nblocks * blockSize, nblocks);
   if (!ReadBlockMat (&src, *transMat, binary))
      HError (9999, "LoadSemiTrans: cannot read transform matrix");
      
   CloseSource(&src);
}

void FVTransModels (HMMSet *hset, BlockMatrix transMat)
{
   int i = 0;
   HMMScanState hss;

   NewHMMScan (hset, &hss);
   do{
      while(GoNextMix (&hss, FALSE)){
         MultBlockMat_Vec (transMat, hss.mp->mean, hss.mp->mean);
         ++i;
      }
   }while (GoNextHMM (&hss));
   EndHMMScan (&hss);

   if (trace & T_ADP)
      fprintf (LOG_STREAM, "applied full-var transform to %d mixture means", i);
}


/* UpdateSpkrModels

     apply speaker specific transforms
*/
Boolean UpdateSpkrModels (char *fn)
{
   char spkrName[MAXSTRLEN] = "";
   char fvTransFN[MAXSTRLEN] = "";
   char mllrTransFN[MAXSTRLEN] = "";
   Boolean changed = FALSE;
   
   /* full-variance transform: apply to means & feature space */
   if (!MaskMatch (spkrPat, spkrName, fn))
      HError (9999, "UpdateSpkrModels: non-matching speaker mask '%s'", spkrPat);
   
   if (!curSpkrName || strcmp (spkrName, curSpkrName)) {
      if (trace & T_ADP)
         fprintf (LOG_STREAM, "new speaker %s, adapting...\n", spkrName);

      /* MLLR transform */
      if (mllrTransDir) {
         if (curSpkrName) {
            /* apply back tranform */
            HError (9999, "UpdateSpkrModels: switching speakers not supported, yet!");
         }
         if (trace & T_ADP)
            fprintf (LOG_STREAM, " applying MLLR transform");

         MakeFN (spkrName, mllrTransDir, NULL, mllrTransFN);
         LoadLegacyTransformSet (&hset, mllrTransFN, rt);

         ApplyTransforms (rt);
         changed = TRUE;
      }

      if (fvTransDir) {
         if (trace & T_ADP)
            fprintf (LOG_STREAM, " applying full-var transform");

         /* full-variance transform */
         if (fvTransMat)
            ResetFVTrans (&hset, fvTransMat);
         MakeFN (spkrName, fvTransDir, NULL, fvTransFN);
         LoadFVTrans (fvTransFN, &fvTransMat);
         FVTransModels (&hset, fvTransMat);
         /* # store per frame offset log(BlkMatDet(transMat)) */

         changed = TRUE;
      }
      curSpkrName = spkrName;
   }

   return changed;
}
#else
Boolean UpdateSpkrModels (char *fn)
{
   HError (1, "MLLR or FV transforms not supported");
   return FALSE;
}
#endif


/*  CC-mode style info for emacs
 Local Variables:
 c-file-style: "htk"
 End:
*/
