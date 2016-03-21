#include<fstream>
#include<iostream>
#include "util.h"
#include "taxonomy_sgd.h"
#include <unordered_set>

using namespace std;

TaxonomyProblem_SGD::TaxonomyProblem_SGD(char* data_file){
    L = 0;
    K = 0;
    N = 0;
    root = new catagory();//root id is 0
    //root.id = 0;
    root->parent = NULL;
    //root->startInd = 0;
    readProblem(data_file);
}

void TaxonomyProblem_SGD::readProblem(char* data_file){
    //read data
    char _line[MAX_LINE];
    string line(_line);
    //vector<Sample> data;
    data.clear();
    Sample ins;
    unordered_set<int> validLabel;
    ifstream fin(data_file);
    int fea;
    double val;
    vector<string> tokens;
    int lbl;
    vector<string> fvpair;
    set<int> feaId;
    while(!fin.eof()){
        fin.getline(_line,MAX_LINE);
        string line(_line);
        split(line," ",tokens);
        if (tokens.size() == 0 )
            continue;
        lbl =atoi(tokens[0].c_str()); 
        labels.push_back(lbl);
        validLabel.insert(lbl);
        ins.clear();
        for (int i=1;i<tokens.size();i++){
            split(tokens[i],":",fvpair);
            fea = atoi(fvpair[0].c_str());
            val = atof(fvpair[1].c_str());
            ins.push_back(make_pair(fea,val));
            feaId.insert(fea);
        }
        data.push_back(ins);
    }
    L = validLabel.size();
    cerr<<"validlabel="<<validLabel.size()<<endl;
    raw_d = feaId.size();
    HashTable<int,int> fea_table(2*raw_d);
    int fid=0;
    for (set<int>::iterator ii = feaId.begin();ii!=feaId.end();++ii){
        fea_table.insert(*ii,fid++);
    }
   //transfer the raw id of feature to compressed id 
    for (vector<Sample>::iterator ii = data.begin(); ii!= data.end();++ii){
        for (Sample::iterator jj = ii->begin(); jj != ii->end();++jj){
           
            fid = *fea_table.findval(jj->first);
            //cerr<<jj->first<<" jj "<<fid<<endl;
            jj->first = fid;
//            jj->first = jj->first-1;
        }
    }

    
    //construct hiearchical tree
    if (param.info_file == NULL){
        cerr<<"exit: Taxonomy Problem needs hierarchy information about the categories"<<endl;
        exit(0);
    }

    ifstream finfo(param.info_file);
    
       //find the number of leaves L by go through the file
   /* while (std::getline(finfo, line))
        L ++;
    finfo.clear();
    finfo.seekg(0, finfo.beg);
*/
    //catagory table whose key is node id and whose value is its catagory point.
    HashTable<int,catagory*> cat_table(3*L);
    int childId;
    cat_table.insert(0,root);// root id is zero. root is not provided by info_file.

    pair<int,catagory*>* rootPair = cat_table.find(0); 
    pair<int,catagory*>* childPair;
    pair<int,catagory*>* parPair;
    K = 1; 

    //pickout the leaves and write it to file such that this model doesn't have hierachical structure. 
    //Actually it will be maximum entropy model. used to compare with GLMNET
//     FILE *fnh = fopen("nohiertest.txt","w");
    int potentialLeaf;
    while (!finfo.eof()){
        finfo.getline(_line,MAX_LINE);
        string line(_line);

        split(line," ",tokens);
        potentialLeaf = atoi((tokens.back()).c_str());
        if (tokens.size()==0 || validLabel.find(potentialLeaf)== validLabel.end()) continue;
        parPair = rootPair;
        for (vector<string>::iterator i=tokens.begin();i!= tokens.end();++i){
            childId = atoi(i->c_str());
            if (( childPair=cat_table.find(childId) )==NULL){
                catagory *childAddr = new catagory();
                K++;
                childAddr->isleaf = false;
                cat_table.insert(childId,childAddr);
                childAddr->parent = parPair->second;
                (parPair->second)->addChild(childAddr);
                parPair = cat_table.find(childId);
            }
            else{ 
                parPair = childPair;
            }
        }
  //      fprintf(fnh,"%d\n",childId);
        (parPair->second)->isleaf = true;
    }
   // fclose(fnh);
  
    N = data.size();
    d = K * raw_d;
    w = new double[d];
    for (int i=0;i<d;i++)
        w[i] = 0.0;
    n = N * K; 
    fvalue = new double[n];
    upMesg = new double[n];
    downMesg = new double[n];
    fv_change_table = new double[n];
    for (int i = 0;i<n;i++){
        fvalue[i] = 0.0;
        upMesg[i] = 0.0;
        downMesg[i] = 0.0;
        fv_change_table[i] = 0.0;
    }
    
    cerr<<"d "<<d<<endl;
    //dataToFeatures(data,raw_d,features); 

    //cerr<<"features "<<features.size()<<endl;
    //initialize starting index for catagory node 
    cerr<<"number of leaves "<<L<<endl;     
    cerr<<"number of catagory "<<K<<endl;
    cerr<<"number of samples"<<N<<endl; 
    //build kinda preorder catagory array
    catArray = new catagory*[K];
    nonleafInd = 0;
    leafInd = K-L;
    root->parInd = -1;
    cerr<<"begin to build"<<endl;
    buildCatArray(root);
    cerr<<"build catArray succussfully"<<endl; 
    //label bitmap
    labelMesg = new int[n];
    for (int i=0;i<n;i++)
        labelMesg[i] = 0;
    cerr<<"number of factors "<<n<<endl;

    catagory* curCat=NULL;
    int nn,labelInd;
    cerr<<"raws label first data: "<<labels[0]<<endl;
    for (int i=0;i<N;i++){
        labelInd = (*cat_table.findval(labels[i]))->ind;
        labels[i] = labelInd;
    }
    for (int i=0;i<N;i++){
        nn = i*K;
        labelInd = labels[i];
        
        labelMesg[nn+labelInd] = 1;
        curCat = catArray[labelInd];
        while((labelInd = curCat->parInd) > -1){
            labelMesg[nn+labelInd] = 1;
            curCat = curCat->parent;//catArray[curCat->parInd];
        }
    }
    infer();
    
    cerr<<"labelMesg sucessful"<<endl;
   
}

void TaxonomyProblem_SGD::buildCatArray(catagory* curCat){
    //i is used to index the nonleaf node while l is used to index leaf
    if (curCat->isleaf == false){
        catArray[nonleafInd] = curCat;
        curCat->ind = nonleafInd++;
        for (vector<catagory*>::iterator ii = curCat->childrenAddrArray.begin(); ii != curCat->childrenAddrArray.end();ii++){
            curCat->childInd.push_back(nonleafInd);
            (*ii)->parInd = curCat->ind;
            buildCatArray(*ii);
        }
    }
    else{
        catArray[leafInd] = curCat;
        curCat->ind = leafInd++;
    }
}

void TaxonomyProblem_SGD::applyGrad(int data_index, double eta, SGD* sgd){
    //compute fvalue of data_index
    int nn = data_index * K;
    int i,f,parentOfl;
    Sample* ins = &(data[data_index]);
    vector<pair<int,double> >::iterator it; 
    int offset;
    for (int j = 0;j<K;j++){
        i = j+nn;
        fvalue[i]=0.0;
        for (it = ins->begin();it!= ins->end();++it){
            f = it->first+raw_d*j;
            fvalue[i] += w[f] * it->second;
        }
    }
    //down message pass
    downMesg[nn] = fvalue[nn];
    for (int j=1;j<K;j++){
        i = nn+j;
        parentOfl = (catArray[j])->parInd;
        downMesg[i] = fvalue[i]+downMesg[nn+parentOfl];
    }

    //normalize
    expOverSumExp(&downMesg[nn+K-L],&upMesg[nn+K-L],L);

    //up message pass
    upMesgPass(root,nn);
    for (int j = 0;j < K;j++){
        offset = j*raw_d;
        i = nn+j;
        //fvalue[i]=0.0;
        for (it = ins->begin();it != ins->end(); ++it){
            f = offset+it->first;
//cerr<<"f="<<f<<" g="<<(upMesg[nn+j] - labelMesg[nn+j])*it->second<<endl;

            w[f] -= eta*(upMesg[i] - labelMesg[i])*it->second;
            sgd->penalty(f,w);
          //  fvalue[i] += w[f] * it->second;
        }
    }
}
/*
void TaxonomyProblem_SGD::compute_fv_change(double* w_change, vector<int>& act_set, //input
		vector<pair<int,double> >& fv_change) //output
{
    int j,c,i,f;
    Feature* feature;
    vector<pair<int,double> >::iterator it;
    for (vector<int>::iterator jj = act_set.begin();jj != act_set.end();++jj){
        j = *jj;//index in w_change
        c = j/raw_d;//corresponding catagory for j index
        f = j%raw_d;//corresponding feature position 
        feature  = &(features[f]);
        for (it = feature->begin();it != feature->end();++it){
            i = it->first * K + c;
            fv_change_table[i] += it->second *w_change[j]; 
        }
        
    }
    //copy to vector
    fv_change.clear();
    for(int i=0;i<n;i++)
        if(fv_change_table[i] != 0.0){
            fv_change.push_back(make_pair(i,fv_change_table[i]));
            fv_change_table[i] = 0.0;
        }
}

void  TaxonomyProblem_SGD::update_fvalue(vector<pair<int,double> >& fv_change, double scalar){
	vector<pair<int,double> >::iterator it;
	for(it=fv_change.begin();it!=fv_change.end(); it++){
		fvalue[it->first] += scalar * it->second;
	}
	infer();
}
*/


double TaxonomyProblem_SGD::upMesgPass(catagory* curCat,int nn){
    if (curCat->isleaf == true)
        return upMesg[nn+curCat->ind];
    else{
        double mesgChildren = 0.0;
        for (vector<catagory*>::iterator ii = curCat->childrenAddrArray.begin();ii != curCat->childrenAddrArray.end();ii++){
            mesgChildren += upMesgPass(*ii,nn);
        }
        upMesg[nn+curCat->ind] = mesgChildren;
        return mesgChildren;
    }
}

void TaxonomyProblem_SGD::infer(){
    int l,parentOfl,nn;
    //int pred;
    //int wrongLabel = 0;
    for (int i=0;i<N;i++){
        nn = i*K;
        downMesg[nn] = fvalue[nn];
        for (int j=1;j<K;j++){
            l = nn+j;
            parentOfl = (catArray[j])->parInd;
            downMesg[l] = fvalue[l]+downMesg[nn+parentOfl];
        }
        //cerr<<"downMesg="<<downMesg[0]<<" "<<downMesg[1]<<endl;
        expOverSumExp(&downMesg[nn+K-L],&upMesg[nn+K-L],L);
      //  if (pred+K-L != labels[i])
        //    wrongLabel++;
        upMesgPass(root,nn); 
    }
    //errorrate = (double)wrongLabel/N;
    //cerr<<"root upmesg value for the first data =  "<<upMesg[0]<<endl;
}

/*
void  TaxonomyProblem_SGD::grad( vector<int>& act_set, //input
			vector<double>& g){//output
    int c,f,i;
    double gii;
    Feature* feature;
    g.clear();
    vector<pair<int,double> >::iterator it;
    for (vector<int>::iterator ii = act_set.begin();ii != act_set.end();ii++){
        f = *ii % raw_d;//raw feature index
        c = *ii / raw_d; //catagory index
        feature = &(features[f]);
        gii = 0.0;
        for (it= feature->begin(); it != feature->end();it++){
            i = it->first;
            gii += (upMesg[i*K+c] - labelMesg[i*K+c])*it->second;
        }
        g.push_back(gii);
    }
}
*/

double  TaxonomyProblem_SGD::fun(){
    int nn,i,f;
    for (int data_index=0;data_index < N;data_index++){
        nn = data_index * K;
        Sample* ins = &(data[data_index]);
        vector<pair<int,double> >::iterator it; 
        for (int j = 0;j<K;j++){
            i = j+nn;
            fvalue[i]=0.0;
            for (it = ins->begin();it!= ins->end();++it){
                f = it->first+raw_d*j;
                fvalue[i] += w[f] * it->second;
            }
        }
    }    
    infer();
    double funvalue = 0.0;
    for (int i=0;i<N;i++){
        nn = i*K;
        funvalue -= log(upMesg[nn+labels[i]]);
    }
    return funvalue;
}
/*
void TaxonomyProblem_SGD::update_Mesg(double *w_change, int f){
    int j,c,i;
    Feature* feature;
    vector<pair<int,double> >::iterator it;
    for (c = 0;c<K;c++){
        j = c*raw_d+f;//index in w_change
        feature  = &(features[f]);
        for (it = feature->begin();it != feature->end();++it){
            i = it->first * K + c;
            fvalue[i] += it->second *w_change[j]; 
        }
    }
    infer(f);
};


void TaxonomyProblem_SGD::infer(int f){
    int l,parentOfl,nn;
    //int wrongLabel = 0;
    Feature* feature = &(features[f]);
    vector<pair<int,double> >::iterator it;
    for (it = feature->begin();it != feature->end();++it){
        nn = it->first*K;
        downMesg[nn] = fvalue[nn];
        for (int j=1;j<K;j++){
            l = nn+j;
            parentOfl = (catArray[j])->parInd;
            downMesg[l] = fvalue[l]+downMesg[nn+parentOfl];
        }
        expOverSumExp(&downMesg[nn+K-L],&upMesg[nn+K-L],L);
     */ /*   if (pred+K-L != labels[it-first])
            wrongLabel++;
      *//*
        upMesgPass(root,nn);
    }
    //errorrate = (double)wrongLabel/N;
    //cerr<<"root upMesg value for the first data =  "<<upMesg[0]<<endl;
}
*/

//double TaxonomyProblem_SGD::firstDerivative(int coordinate){};
	
//double TaxonomyProblem_SGD::secondDerivative(int coordinate){};

