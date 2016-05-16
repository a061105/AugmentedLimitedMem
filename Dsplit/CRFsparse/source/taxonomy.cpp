#include<fstream>
#include<iostream>
#include "util.h"
#include "taxonomy.h"

using namespace std;

TaxonomyProblem::TaxonomyProblem(char* data_file){
    L = 0;
    K = 0;
    N = 0;
    root = new category();
    root->parent = NULL;
    root->raw_label = 0;
    readProblem(data_file);
}
TaxonomyProblem::TaxonomyProblem(char* model_file, char* data_file){
    L = 0;
    K = 0;
    N = 0;
    root = new category();
    root->parent = NULL;
    root->raw_label = 0;
    readProblem(model_file,data_file);

}

void TaxonomyProblem::readProblem(char* model_file, char* data_file){
	readProblem(data_file);
    	ifstream fin(model_file);
    if (fin.fail()){
        cerr<< "can't open model file."<<endl;
        exit(0);
    }

    vector<Int> act_set;
    char* _line = new char[MAX_LINE];
    fin.getline(_line,MAX_LINE);
     
    vector<string> tokens;
    string line_raw_d(_line);
    split(line_raw_d,":",tokens);
    
    Int train_raw_d = atoi(tokens[1].c_str());
  

    Int train_fea,f,c,fea;
    double weight;

    vector<string> fvpair;
    vector<Int> iters;
    act_set.clear();
    fin.getline(_line,MAX_LINE);
    string line(_line);
    split(line," ",tokens);
    for (Int i=0;i<tokens.size();i++){
        split(tokens[i],":",fvpair);
        train_fea = atoi(fvpair[0].c_str());
        c = train_fea / train_raw_d;
        f = train_fea % train_raw_d;
        weight = atof(fvpair[1].c_str());
        if (f < raw_d){
            fea = c * raw_d + f;
            act_set.push_back(fea);
            w[fea] = weight;
        }
    }
    vector<pair<Int,double> > fv_change;
    compute_fv_change(w,act_set,fv_change);
    update_fvalue(fv_change,1.0);

    fin.close();
    delete[] _line;
}

void TaxonomyProblem::readProblem(char* data_file){
    
    //read data
    char* _line = new char[MAX_LINE];
    string line(_line);
    vector<Sample> data;
    Sample ins;
    ifstream fin(data_file);
    if (fin.fail()){
        cerr<< "can't open data file."<<endl;
        exit(0);
    }
    Int fea;
    double val;
    vector<string> tokens;
    Int lbl;
    vector<string> fvpair;
    raw_d = 0; 
    while(!fin.eof()){
    
        fin.getline(_line,MAX_LINE);
        string line(_line);
        split(line," ",tokens);
        if (tokens.size() == 0 )
            continue;
        lbl =atoi(tokens[0].c_str()); 
        labels.push_back(lbl);
        ins.clear();
        for (Int i=1;i<tokens.size();i++){
            split(tokens[i],":",fvpair);
            fea = atoi(fvpair[0].c_str())-1;
            val = atof(fvpair[1].c_str());
            ins.push_back(make_pair(fea,val));
            if (fea>raw_d)
                raw_d = fea;
        }
        data.push_back(ins);
    }
    raw_d ++;
    

    //construct hiearchical tree
    if (param.info_file == NULL){
        cerr<<"exit: Taxonomy Problem needs hierarchy information about the categories"<<endl;
        exit(0);
    }

    ifstream finfo(param.info_file);
     if (finfo.fail()){
        cerr<< "can't open info file."<<endl;
        exit(0);
    }


    while (!finfo.eof()){
        finfo.getline(_line,MAX_LINE);
        string line(_line);

        split(line," ",tokens);
        if (tokens.size() == 0) continue;
        L++;
    }
  
    finfo.clear();
    finfo.seekg(0,ios::beg);
    //category table whose key is node id and whose value is its category poInt.
    HashTable<Int,category*> cat_table(10*L);
    Int childId;
    cat_table.insert(0,root);// root id is zero. root is not provided by info_file.

    pair<Int,category*> rootPair = *cat_table.find(0); 
    pair<Int,category*> childPair;
    pair<Int,category*> parPair;
    K = 1; 

    //Int potentialLeaf;
    while (!finfo.eof()){
        finfo.getline(_line,MAX_LINE);
        string line(_line);

        split(line," ",tokens);
        if (tokens.size() == 0) continue;
        parPair = rootPair;
        for (vector<string>::iterator i=tokens.begin();i!= tokens.end();++i){
            childId = atoi(i->c_str());
            if ( cat_table.find(childId) ==NULL){
                category *childAddr = new category();
                K++;
                childAddr->isleaf = false;
                childAddr->raw_label = childId;
                cat_table.insert(childId,childAddr);
                childAddr->parent = parPair.second;
                (parPair.second)->addChild(childAddr);
                parPair = *cat_table.find(childId);
            }
            else{ 
                parPair = *cat_table.find(childId) ;
            }
        }
        (parPair.second)->isleaf = true;
    }
    N = data.size();
    d = K * raw_d;
    w = new double[d];
    for (Int i=0;i<d;i++)
        w[i] = 0.0;
    n = N * K;
    fvalue = new double[n];
    upMesg = new double[n];
    downMesg = new double[n];
    fv_change_table = new double[n];
    for (Int i = 0;i<n;i++){
        fvalue[i] = 0.0;
        upMesg[i] = 0.0;
        downMesg[i] = 0.0;
        fv_change_table[i] = 0.0;
    }
    
    //transfer the data-indexed structure to feature-indexed structure
    dataToFeatures(data,raw_d,features); 
    //build a preorder category array
    catArray = new category*[K];
    nonleafInd = 0;
    leafInd = K-L;
    root->parInd = -1;
    buildCatArray(root);

    //label bitmap
    labelMesg = new double[n];
    for (Int i=0;i<n;i++)
        labelMesg[i] = 0;

    category* curCat=NULL;
    Int nn,labelInd;
    for (Int i=0;i<N;i++){
        labelInd = (*cat_table.findval(labels[i]))->ind;
        labels[i] = labelInd;
    }
    for (Int i=0;i<N;i++){
        nn = i*K;
        labelInd = labels[i];
        labelMesg[nn+labelInd] = 1.0;
        curCat = catArray[labelInd];
        while((labelInd = curCat->parInd) > -1){
            labelMesg[nn+labelInd] = 1.0;
            curCat = curCat->parent;
        }
    }

    delete[] _line;
    infer();
       
}

void TaxonomyProblem::buildCatArray(category* curCat){

    //i is used to index the nonleaf node while l is used to index leaf
    if (curCat->isleaf == false){
        catArray[nonleafInd] = curCat;
        curCat->ind = nonleafInd++;
        for (vector<category*>::iterator ii = curCat->childrenAddrArray.begin(); ii != curCat->childrenAddrArray.end();ii++){
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

void TaxonomyProblem::compute_fv_change(double* w_change, vector<Int>& act_set, //input
		vector<pair<Int,double> >& fv_change) //output
{
    Int j,c,i,f;
    Feature* feature;

    vector<pair<Int,double> >::iterator it;
    for (vector<Int>::iterator jj = act_set.begin();jj != act_set.end();++jj){
        j = *jj;//index in w_change
        c = j/raw_d;//corresponding category for j index
        f = j%raw_d;//corresponding feature position 
        feature  = &(features[f]);
        for (it = feature->begin();it != feature->end();++it){
            i = it->first * K + c;
            fv_change_table[i] += it->second *w_change[j]; 
        }
        
    }
    //copy to vector
    fv_change.clear();
    for(Int i=0;i<n;i++)
        if(fv_change_table[i] != 0.0){
            fv_change.push_back(make_pair(i,fv_change_table[i]));
            fv_change_table[i] = 0.0;
        }
}

void  TaxonomyProblem::update_fvalue(vector<pair<Int,double> >& fv_change, double scalar){
	vector<pair<Int,double> >::iterator it;
	for(it=fv_change.begin();it!=fv_change.end(); it++){
		fvalue[it->first] += scalar * it->second;
	}
	infer();
}

double TaxonomyProblem::upMesgPass(category* curCat,Int nn){
    if (curCat->isleaf == true)
        return upMesg[nn+curCat->ind];
    else{
        double mesgChildren = 0.0;
        for (vector<category*>::iterator ii = curCat->childrenAddrArray.begin();ii != curCat->childrenAddrArray.end();ii++){
            mesgChildren += upMesgPass(*ii,nn);
        }
        upMesg[nn+curCat->ind] = mesgChildren;
        return mesgChildren;
    }
}

void TaxonomyProblem::infer(){
    Int l,parentOfl,nn;
    Int pred;
    for (Int i=0;i<N;i++){
        nn = i*K;
        downMesg[nn] = fvalue[nn];
        for (Int j=1;j<K;j++){
            l = nn+j;
            parentOfl = (catArray[j])->parInd;
            downMesg[l] = fvalue[l]+downMesg[nn+parentOfl];
        }
        pred = expOverSumExp(&downMesg[nn+K-L],&upMesg[nn+K-L],L);
        upMesgPass(root,nn); 
    }
}

void  TaxonomyProblem::grad( vector<Int>& act_set, //input
			vector<double>& g){//output 
    Int c,f,i;
    double gii;
    Feature* feature;
    g.clear();
    vector<pair<Int,double> >::iterator it;
    for (vector<Int>::iterator ii = act_set.begin();ii != act_set.end();ii++){
        f = *ii % raw_d;//raw feature index
        c = *ii / raw_d; //category index
        feature = &(features[f]);
        gii = 0.0;
        
        for (it= feature->begin(); it != feature->end();it++){
            i = it->first;
            gii += (upMesg[i*K+c] - labelMesg[i*K+c])*it->second;
        }
        g.push_back(gii);
    }
}

double  TaxonomyProblem::fun(){
    double funvalue = 0.0;
    Int nn;
    for (Int i=0;i<N;i++){
        nn = i*K;
        funvalue -= log(upMesg[nn+labels[i]]);
    }
    return funvalue;
}
double  TaxonomyProblem::fun(vector<Int>& act_set){
    double funvalue = 0.0;
    Int nn;
    for (Int i=0;i<N;i++){
        nn = i*K;
        funvalue -= log(upMesg[nn+labels[i]]);
    }
    return funvalue;
}

double TaxonomyProblem::train_accuracy(){
    double accuracy = 0.0;
    Int nn;
    Int pred;
    for (Int i=0;i<N;i++){
        nn = i*K;
        //error += (1-upMesg[nn + labels[i]]);
        maximum(&downMesg[nn+K-L],L,pred);
        if (K - L + pred == labels[i])
            accuracy ++;
        //pred = expOverSumExp(&downMesg[nn+K-L],&upMesg[nn+K-L],L);
    }
    return accuracy / N;
}


void TaxonomyProblem::test_accuracy(const char* output_file){
    ofstream fout(output_file);
    infer();

    double accuracy = 0.0;
    Int nn,pred;
    for (Int i=0;i<N;i++){
        nn = i*K;
        maximum(&downMesg[nn+K-L],L,pred);
        fout << catArray[K-L+pred]->raw_label << endl;
        if (K - L + pred == labels[i])
            accuracy ++;
    }

    cerr<<"testing accuracy: "<<accuracy / N<<endl;
    
    fout.close();
}

