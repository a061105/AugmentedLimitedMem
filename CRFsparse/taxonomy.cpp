#include<fstream>
#include<iostream>
#include "util.h"
#include "taxonomy.h"
#include <unordered_set>

using namespace std;

TaxonomyProblem::TaxonomyProblem(char* data_file, char* info_file){
    L = 0;
    K = 0;
    N = 0;
    root = new catagory();//root id is 0
    //root.id = 0;
    root->parent = NULL;
    //root->startInd = 0;
    readProblem(data_file, info_file);
}

void TaxonomyProblem::readProblem(char* data_file, char* info_file){
    //read data
    char _line[MAX_LINE];
    string line(_line);
    vector<Sample> data;
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
    //cerr<<"validlabel="<<validLabel.size()<<endl;
    raw_d = feaId.size();
    cerr<<"raw_d="<<raw_d<<endl;
    numBlocks = raw_d;
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
    if (info_file == NULL){
        cerr<<"exit: Taxonomy Problem needs hierarchy information about the categories"<<endl;
        exit(0);
    }

    ifstream finfo(info_file);
    
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

    pair<int,catagory*> rootPair = *cat_table.find(0); 
    pair<int,catagory*> childPair;
    pair<int,catagory*> parPair;
    K = 1; 

    //pickout the leaves and write it to file such that this model doesn't have hierachical structure. 
    //Actually it will be maximum entropy model. used to compare with GLMNET
//     FILE *fnh = fopen("nohiertest.txt","w");
    int potentialLeaf;
    while (!finfo.eof()){
        finfo.getline(_line,MAX_LINE);
        string line(_line);

        split(line," ",tokens);
        if (tokens.size() == 0) continue;
        potentialLeaf = atoi((tokens.back()).c_str());
        if (validLabel.find(potentialLeaf)== validLabel.end()) continue;
        parPair = rootPair;
        for (vector<string>::iterator i=tokens.begin();i!= tokens.end();++i){
            childId = atoi(i->c_str());
            if ( cat_table.find(childId) ==NULL){
                catagory *childAddr = new catagory();
                K++;
                childAddr->isleaf = false;
                cat_table.insert(childId,childAddr);
                childAddr->parent = parPair.second;
                //root->addChild(childAddr);
                (parPair.second)->addChild(childAddr);
                parPair = *cat_table.find(childId);
            }
            else{ 
                parPair = *cat_table.find(childId) ;
            }
        }
  //      fprintf(fnh,"%d\n",childId);
        (parPair.second)->isleaf = true;
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
    
    //cerr<<"d "<<d<<endl;
    dataToFeatures(data,raw_d,features); 

    cerr<<"features "<<features.size()<<endl;
    //initialize starting index for catagory node 
    cerr<<"number of leaves "<<L<<endl;     
    cerr<<"number of catagory "<<K<<endl;
    cerr<<"number of samples"<<N<<endl; 
    //build kinda preorder catagory array
    catArray = new catagory*[K];
    nonleafInd = 0;
    leafInd = K-L;
    root->parInd = -1;
    //cerr<<"begin to build"<<endl;
    buildCatArray(root);
 
    //  for (int i=0;i<K;i++){
//        cerr<< catArray[i]->isleaf<<endl;
    //}
    //cerr<<"build catArray succussfully"<<endl; 
    //label bitmap
    labelMesg = new double[n];
    for (int i=0;i<n;i++)
        labelMesg[i] = 0.0;

    catagory* curCat=NULL;
    int nn,labelInd;
    //cerr<<"raws label first data: "<<labels[0]<<endl;
    for (int i=0;i<N;i++){
        labelInd = (*cat_table.findval(labels[i]))->ind;
        labels[i] = labelInd;
    }
    for (int i=0;i<N;i++){
        nn = i*K;
        labelInd = labels[i];
        
        labelMesg[nn+labelInd] = 1.0;
        curCat = catArray[labelInd];
        while((labelInd = curCat->parInd) > -1){
            labelMesg[nn+labelInd] = 1.0;
            curCat = curCat->parent;//catArray[curCat->parInd];
        }
    }
    infer();
    
    //cerr<<"labelMesg sucessful"<<endl;
   
    /*
    //transform the data to CRF++ format input file. 
    for (int i=0;i<N;i++)
        data[i].clear();
    for (int i=0;i<raw_d;i++){
        vector<pair<int,double> >* ins = &(features[i]);
    	vector<pair<int,double> >::iterator it;
		for(it=ins->begin();it!=ins->end();it++){
			data[it->first].push_back(make_pair(i,it->second));
		}
	}
  cerr<<data[0].size()<<" check "<<data[1].size()<<endl;
  int chk=0;
  int common=0;
    for( 	vector<pair<int,double> >::iterator it = data[0].begin();it != data[0].end();){
        if (it->first > data[1][chk].first)
            chk++;
        else if(it->first < data[1][chk].first)
            it++;
        else{
            if (it->second != data[1][chk].second)
               // common++;
            it++;
            chk++;
        }

    }
//cerr<<"common "<<common<<endl;
    FILE *fi = fopen("train.50.glmnet","w");
    vector<int>::iterator l = labels.begin();
	for (vector<Sample>::iterator ii= data.begin();ii!=data.end();++ii)
    {
        int a=0;
        Sample::iterator ins = ii->begin();
        for (int i=0;i<raw_d;i++){
            if (ins->first == i && ins!=ii->end()){
                fprintf(fi,"%d ",(int)ins->second);
                ins++;
            }
            else
                fprintf(fi,"%d ",0);
        }
        fprintf(fi,"%d \n",*l);
        l++;
    }
    fclose(fi);
    exit(0);
    */
    
}


void TaxonomyProblem::buildCatArray(catagory* curCat){
    //i is used to index the nonleaf node while l is used to index leaf
    if (curCat->isleaf == false){
        catArray[nonleafInd] = curCat;
        curCat->ind = nonleafInd++;
        //cerr<<curCat->ind<<" ";
        //cerr<<"nonleafInd "<<nonleafInd<<endl;
        for (vector<catagory*>::iterator ii = curCat->childrenAddrArray.begin(); ii != curCat->childrenAddrArray.end();ii++){
            curCat->childInd.push_back(nonleafInd);
            (*ii)->parInd = curCat->ind;
            buildCatArray(*ii);
        }
    }
    else{
        catArray[leafInd] = curCat;
        curCat->ind = leafInd++;
      //  cerr<<"leafInd "<<leafInd<<endl;
    }
    //cerr<<"leafInd end "<<leafInd<<"; nonleafInd end= "<<nonleafInd<<endl;
}

void TaxonomyProblem::compute_fv_change(double* w_change, vector<int>& act_set, //input
		vector<pair<int,double> >& fv_change) //output
{
    int j,c,i,f;
    Feature* feature;
/*    for (int l = 0;l<n;l++){
        fv_change_table[l] = 0.0;
    }
*/
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

void  TaxonomyProblem::update_fvalue(vector<pair<int,double> >& fv_change, double scalar){
	vector<pair<int,double> >::iterator it;
	for(it=fv_change.begin();it!=fv_change.end(); it++){
		fvalue[it->first] += scalar * it->second;
	}
    //cerr<<"update fvalue"<<endl;	
	infer();
}

double TaxonomyProblem::upMesgPass(catagory* curCat,int nn){
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

void TaxonomyProblem::infer(){
    int l,parentOfl,nn;
    int pred;
    int wrongLabel = 0;
    for (int i=0;i<N;i++){
        nn = i*K;
        downMesg[nn] = fvalue[nn];
        for (int j=1;j<K;j++){
            l = nn+j;
            parentOfl = (catArray[j])->parInd;
            downMesg[l] = fvalue[l]+downMesg[nn+parentOfl];
        }
        //cerr<<"downMesg="<<downMesg[0]<<" "<<downMesg[1]<<endl;
        pred = expOverSumExp(&downMesg[nn+K-L],&upMesg[nn+K-L],L);
        if (pred+K-L != labels[i])
            wrongLabel++;
        upMesgPass(root,nn); 
    }
    errorrate = (double)wrongLabel/N;
    //cerr<<"root upmesg value for the first data =  "<<upMesg[0]<<endl;
}
/*
//CD derivatives
void  TaxonomyProblem::derivatives( int coordinate, //input
			vector<double>& g, vector<double> &h){//output //tmp
    int c,f,i;
    vector<int> act_set;
    act_set.clear();
    act_set.push_back(coordinate);
    double hii;//tmp
    h.clear();//tmp
    double gii;
    Feature* feature;
    g.clear();
    vector<pair<int,double> >::iterator it;
    for (vector<int>::iterator ii = act_set.begin();ii != act_set.end();ii++){
        f = *ii % raw_d;//raw feature index
        c = *ii / raw_d; //catagory index
        feature = &(features[f]);
        gii = 0.0;
        
        hii=0.0;//tmp
        
        for (it= feature->begin(); it != feature->end();it++){
            i = it->first;
            gii += (upMesg[i*K+c] - labelMesg[i*K+c])*it->second;

            hii +=upMesg[i*K+c]*(1.0-upMesg[i*K+c] ) *it->second*it->second; //tmp 


        }
        h.push_back(hii);
        g.push_back(gii);
    }
}
*/

void  TaxonomyProblem::grad( vector<int>& act_set, //input
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


double  TaxonomyProblem::fun(){
    double funvalue = 0.0;
    int nn;
    //double aveErrorRate=0.0;
    for (int i=0;i<N;i++){
        nn = i*K;
        funvalue -= log(upMesg[nn+labels[i]]);
    }
    return funvalue;
}
/*
//CD update_Mesg
void TaxonomyProblem::update_Mesg(int j,double* delta_w){
    int f,c,i,nn;
    Feature* feature;
    vector<pair<int,double> >::iterator it;
    f = j % raw_d;
    c = j / raw_d;
    feature  = &(features[f]);
    //cerr<<"fvalue:";
    for (it = feature->begin();it != feature->end();++it){
        nn = it->first * K;
        fvalue[nn+c] += it->second*(delta_w[j]);
            //cerr<<fvalue[nn+c]<<" ";
    }
    //cerr<<endl;
    infer(f);
}
*/
// BCD update_Mesg
void TaxonomyProblem::update_Mesg(vector<int> &act_set, double* delta_w){
    if (act_set.size()==0)
        return;
    int f=act_set[0] % raw_d;
    int j,c;
    Feature* feature;
    vector<pair<int,double> >::iterator it;
    feature  = &(features[f]);
    //cerr<<"fvalue:";
    for (vector<int>::iterator ii = act_set.begin();ii != act_set.end();ii++){
        j = *ii;
        c = j / raw_d;
        //for (c = 0;c<K;c++){
      //  j = c*raw_d+f;
        for (it = feature->begin();it != feature->end();++it){
            fvalue[it->first * K+c] += it->second*(delta_w[j]);
            //cerr<<fvalue[nn+c]<<" ";
        }
    }
    //cerr<<endl;
    infer(f);
}

vector<int>* TaxonomyProblem::buildBlocks(){
    vector<int>* blocks = new vector<int>[numBlocks];
    for (int f=0;f<numBlocks;f++)
        for (int c=0;c<K;c++){
            blocks[f].push_back(c * raw_d+f);
        }
    return blocks;
}

void TaxonomyProblem::infer(int f){
    int l,parentOfl,nn;
    //int pred;
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
     /*   if (pred+K-L != labels[it-first])
            wrongLabel++;
      */
        upMesgPass(root,nn);
    }
    //errorrate = (double)wrongLabel/N;
    //cerr<<"root upmesg value for the first data =  "<<upMesg[0]<<endl;
}


//BCD derivatives
void TaxonomyProblem::derivatives(vector<int> &act_set, vector<double> &firstDerivative, vector<double> &secondDerivative){
    firstDerivative.clear();
    secondDerivative.clear();
    int c,i,f;
    double g,h;
    Feature* feature;
    vector<pair<int,double> >::iterator it;
    for (vector<int>::iterator ii = act_set.begin();ii != act_set.end();ii++){
    //for (c = 0;c<K;c++){
        f = *ii % raw_d;//raw feature index
        c = *ii / raw_d; //catagory index
        feature = &(features[f]);
        g = 0.0;
        h = 0.0;
        for (it= feature->begin(); it != feature->end();it++){
            i = it->first*K+c;
            g += (upMesg[i] - labelMesg[i])*it->second;
            h += upMesg[i]*(1.0-upMesg[i]) * it->second * it->second;
            
        }
        firstDerivative.push_back(g);
        secondDerivative.push_back(h);
    }
}
		
void TaxonomyProblem::save(char* filename){
	
}

void TaxonomyProblem::load(char* filename){
	
}

void TaxonomyProblem::compute_fvalue(){

}
