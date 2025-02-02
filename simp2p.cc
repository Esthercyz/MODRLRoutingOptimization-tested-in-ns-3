#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <map>
#include <utility>
#include <set>

#include <stdio.h>
#include <stdlib.h>

#include "ns3/core-module.h"
#include "ns3/opengym-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/csma-module.h"
#include "ns3/ipv4-nix-vector-helper.h"
#include "ns3/netanim-module.h"
#include "ns3/constant-position-mobility-model.h"


#include "MO-env.h"

#define TG_CDF_TABLE_ENTRY 32

#define LINK_CAPACITY_BASE   1000000000          // 1Gbps
#define BUFFER_SIZE 250    //250 packets

// The flow port range, each flow will be assigned a random port number within this range

//static uint16_t port = 1000;

#define PACKET_SIZE 1400

using namespace ns3;
using namespace std;



struct cdf_entry
{
    double value;
    double cdf;
};

/* CDF distribution */
struct cdf_table
{
    struct cdf_entry *entries;
    int num_entry;  /* number of entries in CDF table */
    int max_entry;  /* maximum number of entries in CDF table */
    double min_cdf; /* minimum value of CDF (default 0) */
    double max_cdf; /* maximum value of CDF (default 1) */
};

/* initialize a CDF distribution */
void init_cdf(struct cdf_table *table);

/* free resources of a CDF distribution */
void free_cdf(struct cdf_table *table);

/* get CDF distribution from a given file */
void load_cdf(struct cdf_table *table, const char *file_name);

/* print CDF distribution information */
void print_cdf(struct cdf_table *table);

/* get average value of CDF distribution */
double avg_cdf(struct cdf_table *table);

/* Generate a random value based on CDF distribution */
double gen_random_cdf(struct cdf_table *table);

/* initialize a CDF distribution */
void init_cdf(struct cdf_table *table)
{
    if(!table)
        return;

    table->entries = (struct cdf_entry*)malloc(TG_CDF_TABLE_ENTRY * sizeof(struct cdf_entry));
    table->num_entry = 0;
    table->max_entry = TG_CDF_TABLE_ENTRY;
    table->min_cdf = 0;
    table->max_cdf = 1;

    if (!(table->entries))
        perror("Error: malloc entries in init_cdf()");
}

/* free resources of a CDF distribution */
void free_cdf(struct cdf_table *table)
{
    if (table)
        free(table->entries);
}

/* get CDF distribution from a given file */
void load_cdf(struct cdf_table *table, const char *file_name)
{
    FILE *fd = NULL;
    char line[256] = {0};
    struct cdf_entry *e = NULL;
    int i = 0;

    if (!table)
        return;

    fd = fopen(file_name, "r");
    if (!fd)
        perror("Error: open the CDF file in load_cdf()");

    while (fgets(line, sizeof(line), fd))
    {
        /* resize entries */
        if (table->num_entry >= table->max_entry)
        {
            table->max_entry *= 2;
            e = (struct cdf_entry*)malloc(table->max_entry * sizeof(struct cdf_entry));
            if (!e)
                perror("Error: malloc entries in load_cdf()");
            for (i = 0; i < table->num_entry; i++)
                e[i] = table->entries[i];
            free(table->entries);
            table->entries = e;
        }

        sscanf(line, "%lf %lf", &(table->entries[table->num_entry].value), &(table->entries[table->num_entry].cdf));

        if (table->min_cdf > table->entries[table->num_entry].cdf)
            table->min_cdf = table->entries[table->num_entry].cdf;
        if (table->max_cdf < table->entries[table->num_entry].cdf)
            table->max_cdf = table->entries[table->num_entry].cdf;

        table->num_entry++;
    }
    fclose(fd);
}

/* print CDF distribution information */
void print_cdf(struct cdf_table *table)
{
    int i = 0;

    if (!table)
        return;

    for (i = 0; i < table->num_entry; i++)
        printf("%.2f %.2f\n", table->entries[i].value, table->entries[i].cdf);
}

/* get average value of CDF distribution */
double avg_cdf(struct cdf_table *table)
{
    int i = 0;
    double avg = 0;
    double value, prob;

    if (!table)
        return 0;

    for (i = 0; i < table->num_entry; i++)
    {
        if (i == 0)
        {
            value = table->entries[i].value / 2;
            prob = table->entries[i].cdf;
        }
        else
        {
            value = (table->entries[i].value + table->entries[i-1].value) / 2;
            prob = table->entries[i].cdf - table->entries[i-1].cdf;
        }
        avg += (value * prob);
    }
    return avg;
}

double interpolate(double x, double x1, double y1, double x2, double y2)
{
    if (x1 == x2)
        return (y1 + y2) / 2;
    else
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

/* generate a random floating point number from min to max */
double rand_range(double min, double max)
{
    return min + rand() * (max - min) / RAND_MAX;
}

/* generate a random value based on CDF distribution */
double gen_random_cdf(struct cdf_table *table)
{
    int i = 0;
    double x = rand_range(table->min_cdf, table->max_cdf);
    /* printf("%f %f %f\n", x, table->min_cdf, table->max_cdf); */

    if (!table)
        return 0;

    for (i = 0; i < table->num_entry; i++)
    {
        if (x <= table->entries[i].cdf)
        {
            if (i == 0)
                return interpolate(x, 0, 0, table->entries[i].cdf, table->entries[i].value);
            else
                return interpolate(x, table->entries[i-1].cdf, table->entries[i-1].value, table->entries[i].cdf, table->entries[i].value);
        }
    }

    return table->entries[table->num_entry-1].value;
}

NS_LOG_COMPONENT_DEFINE("fattreeSimp2p");


template<typename T>
T rand_range(T min, T max)
{
    return min+((double)max-min)*rand()/ RAND_MAX;
}

void printRoutingTable (Ptr<Node> node) 
{
        Ipv4StaticRoutingHelper helper;
	Ptr<Ipv4> stack = node -> GetObject<Ipv4>();
	Ptr<Ipv4StaticRouting> staticrouting = helper.GetStaticRouting(stack);
        uint32_t numroutes=staticrouting->GetNRoutes();
        Ipv4RoutingTableEntry entry;
        std::cout << "Routing table for device: " << Names::FindName(node) <<"\n";
        std::cout << "Destination\tMask\t\tGateway\t\tIface\n";
        for (uint32_t i =0 ; i<numroutes;i++) {
        entry =staticrouting->GetRoute(i);
        std::cout << entry.GetDestNetwork()  << "\t" << entry.GetDestNetworkMask() << "\t" << entry.GetGateway() << "\t\t" << entry.GetInterface() << "\n";
	}
	  return;
} 

void print_stats ( FlowMonitor::FlowStats st)
{
	cout << " Tx Bytes : " << st.txBytes << endl;
        cout << " Rx Bytes : " << st.rxBytes << endl;
        cout << " Tx Packets : " << st.txPackets << endl;
        cout << " Rx Packets : " << st.rxPackets << endl;
        cout << " Lost Packets : " << st.lostPackets << endl;
        cout << " Dropped bytes : " << st.bytesDropped.size() << endl;

        if( st.rxPackets > 0) {
            cout << " Mean{Delay}: " << (st.delaySum.GetSeconds() / st.rxPackets);
            cout << " Mean{jitter}: " << (st.jitterSum.GetSeconds() / (st.rxPackets -1));
            cout << " Mean{Hop Count }: " << st.timesForwarded / st.rxPackets + 1;
        }
        if (true) {
           cout << "Delay Histogram : " << endl;
           for(uint32_t i = 0; i < st.delayHistogram.GetNBins (); i++)
              cout << " " << i << "(" << st.delayHistogram.GetBinStart (i) << st.delayHistogram.GetBinEnd(i) << "):" << st.delayHistogram.GetBinCount (i) << endl;

        }
}
//and then in the callback, capture number of bytes entering the link.


/*
void CalcStatictis()
{
//compute link utilisation

   for(int i=0;i<36;i++){
      for(int j=0;j<8;j++){
        cout<<total_bytes[i][j]<<'\t';
       }
      cout<<endl;
   }
   
   Simulator::Schedule (Seconds (2), &CalcStatictis);
}
*/

//NS_LOG_COMPONENT_DEFINE ("SimpleGlobalRoutingExample");

int 
main(int argc, char *argv[])
{
  // Users may find it convenient to turn on explicit debugging
  // for selected modules; the below line suggets how to do this
#if 0
  LogComponentEnable ("SimpleGlobalRoutingExample", LOG_LEVEL_INFO);
#endif
  // Set up some default values for the simulation. Use the

  std::string cdfFileName = "/home/ck/software/ns3-gym-master/scratch/MODRL/simp2p_CDF.txt";

  //double START_TIME = 0.5;
  double END_TIME = 2.0;

  uint32_t linkLatency = 10;

  uint64_t switchLinkCapacity = 10;
  uint64_t serverLinkCapacity = 10;

  uint64_t SWITCH_LINK_CAPACITY = switchLinkCapacity * LINK_CAPACITY_BASE;
  uint64_t SERVER_LINK_CAPACITY = serverLinkCapacity * LINK_CAPACITY_BASE;
  Time LINK_LATENCY = MicroSeconds(linkLatency);

  
  bool set_successful = true;
  set_successful &= Config::SetDefaultFailSafe("ns3::Ipv4GlobalRouting::RandomEcmpRouting", BooleanValue(true));
  if (!set_successful) 
  {
    std::cerr << "Error: Could not set defaults" << std::endl;
    return 0;
  }
  
  //Config::SetDefault ("ns3::Ipv4GlobalRouting::PerflowEcmpRouting", BooleanValue(true));

  Config::SetDefault("ns3::Ipv4GlobalRouting::RespondToInterfaceEvents",BooleanValue(true));

   NS_LOG_INFO ("Config parameters");
  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue(PACKET_SIZE));
  Config::SetDefault ("ns3::TcpSocket::DelAckCount", UintegerValue (0));
  Config::SetDefault ("ns3::TcpSocket::ConnTimeout", TimeValue (MilliSeconds (5)));

  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (10));
  Config::SetDefault ("ns3::TcpSocketBase::MinRto", TimeValue (MilliSeconds (5)));
  Config::SetDefault ("ns3::TcpSocketBase::ClockGranularity", TimeValue (MicroSeconds (100)));
  Config::SetDefault ("ns3::RttEstimator::InitialEstimation", TimeValue (MicroSeconds (80)));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (160000000));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (160000000));

  //Config::SetDefault ("ns3::OnOffApplication::PacketSize", UintegerValue(2000));
  //Config::SetDefault ("ns3::OnOffApplication::DataRate", StringValue ("1.5Mb/s"));
  //uint32_t mtu_bytes = 400;

  // DefaultValue::Bind ("DropTailQueue::m_maxPackets", 30);
  // Allow the user to override any of the default and the above
  // DafaultValue::Bind ()s at run-time, via command-line arguments
  int k=4,i,j,temp,l1_index,p;
  uint32_t simSeed = 1;
  double simulationTime = 10;
  double envStepTime = 0.005;
  uint32_t openGymPort = 5555;
  //uint32_t testArg = 0;

  CommandLine cmd;
  bool enableMonitor = true;
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  std::string animFile = "fat-animation.xml";;
  cmd.AddValue("EnableMonitor", "Enable Flow Monitor", enableMonitor);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
  cmd.AddValue("animFile", "Filename for animation output", animFile);
  cmd.Parse(argc, argv);
  Ipv4AddressHelper address;
  char str1[30],str2[30],str3[30],str4[30];

  // For BulkSend
  //uint32_t maxBytes = 52428800; // 50 MB of data

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);

  // OpenGym Env
  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
  Ptr<MOGymEnv> myGymEnv = CreateObject<MOGymEnv> (Seconds(envStepTime));
  myGymEnv->SetOpenGymInterface(openGymInterface);

  NS_LOG_INFO ("Create Nodes");
  // Leaf Level subnets, there are k*(k/2) subnets
  NodeContainer* c = new NodeContainer[16];


// Aggregate routers, there are k groups of routers , and in each group ther are k routers numbered from 0..k-1
  NodeContainer* l1 = new NodeContainer[4];
  // Core routrer, there are (k/2) ^ 2 in numbers.
  NodeContainer l2;
  
  // CsmaHelper for obtaining CSMA behavior in subnet
  CsmaHelper csma;
  // Point to Point links are used to interconnect the PODS
  PointToPointHelper p2p;
  //char* linkRate = (char*)"25Mbps";
  //char* linkDelay = (char*)"25us";
  p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate(SERVER_LINK_CAPACITY)));
  p2p.SetChannelAttribute ("Delay", TimeValue (LINK_LATENCY));
  p2p.SetQueue("ns3::DropTailQueue");
  
  // NetDeviceContainer to aggregate the subnet with the aggregate router, equal to number of subnets
  NetDeviceContainer* d = new NetDeviceContainer[16]; 
  // Also have Ipv4InterfaceContainer,offcourse equal to number of NetDeviceContainer
  Ipv4InterfaceContainer* ip = new Ipv4InterfaceContainer[16];
  // Have point-to-point connection between two set of aggregate routers.
  NetDeviceContainer* p2p_d = new NetDeviceContainer[1024];
  NetDeviceContainer inter_level1;
  // Seperate Ipv4InterfaceContainer for point to point links
  Ipv4InterfaceContainer* ip_for_p2p = new Ipv4InterfaceContainer[1024]; 
  // Have k/2 aggregate router

  // Core level NetDeviceContainers for interconnecting aggregate routers with core routers
  NetDeviceContainer* core_d = new NetDeviceContainer[1024];
  // Core Ipv4InterfaceContainer
  Ipv4InterfaceContainer* ip_for_core = new Ipv4InterfaceContainer[1024];

  // Set starting time
  //clock_t start = clock();
  InternetStackHelper internet;
  Ipv4GlobalRoutingHelper globalRoutingHelper;

  //Ipv4NixVectorHelper nixRouting; 
  //Ipv4StaticRoutingHelper staticRouting;
  //Ipv4ListRoutingHelper list;
  //list.Add (staticRouting, 0);
  //list.Add (nixRouting, 10);
  internet.SetRoutingHelper(globalRoutingHelper);
  
 


  std::pair<Ptr<Ipv4>, uint32_t> pairx; 
  // First create core router
  l2.Create(4);
  internet.Install(l2);
   int subnet_index = 0;
  l1_index = 0;
  
  // For animation
  float pos_x = 60;
  float pos_y = 20;
  float save;
  float save_x;
  // First place the core routers
  for( int x = 0; x < k ; x++){
                Ptr<Node> n1 = l2.Get(x);
               Ptr<ConstantPositionMobilityModel> loc1 = n1->GetObject<ConstantPositionMobilityModel> ();
                if(loc1 == 0) {
                        loc1 = CreateObject<ConstantPositionMobilityModel> ();
                        Vector vec1(pos_x,pos_y,0);
                        loc1->SetPosition(vec1);
                        n1->AggregateObject(loc1);
                }
                pos_x = pos_x + 30;
  }
  printf("Cores are places\n");
  // Placing core ends here
   pos_x = 10;
   pos_y = 70;

  // Constructing k pods 
  for(i=0; i<k; i++) {
        //cout << " *******************Pod " << i <<" ***********************" << endl;
        l1[i].Create(k);
        save = pos_x;
        pos_y = 70;
        for( int x = k/2; x < k; x++){
                Ptr<Node> n1 = l1[i].Get(x);
                Ptr<ConstantPositionMobilityModel> loc1 = n1->GetObject<ConstantPositionMobilityModel> ();
                if(loc1 == 0) {
                        loc1 = CreateObject<ConstantPositionMobilityModel> ();
                        Vector vec1(pos_x,pos_y,0);
                        loc1->SetPosition(vec1);
                        n1->AggregateObject(loc1);
                        //cout << "Pod: " << i << " Subnet: " << subnet_index << "Node:0 " << pos_x << " " << pos_y << endl;
                }
                pos_x = pos_x + 30;
        }
        save_x = save;
        pos_y = 110;
        for( int x = 0; x < k/2 ; x++){
                Ptr<Node> n1 = l1[i].Get(x);
                Ptr<ConstantPositionMobilityModel> loc1 = n1->GetObject<ConstantPositionMobilityModel> ();
                if(loc1 == 0) {
                        loc1 = CreateObject<ConstantPositionMobilityModel> ();
                        Vector vec1(save_x,pos_y,0);
                        loc1->SetPosition(vec1);
                        n1->AggregateObject(loc1);
                        //cout << "Pod: " << i << " Subnet: " << subnet_index << "Node:0 " << pos_x << " " << pos_y << endl;
                }
                save_x = save_x + 30;
         }
        //printf("Aggr placed\n");
	internet.Install(l1[i]);
        save_x = save;
        pos_y = 150;
        for ( j=0; j < k/2 ; j++)
	{
	   temp = (k/2) * i + j;
	   //cout << "Subnet Number :" << temp << endl;
	   c[subnet_index].Create(k/2);
           // Place the nodes at suitable position for Animation
           Ptr<Node> n1 = c[subnet_index].Get(0);
           Ptr<ConstantPositionMobilityModel> loc1 = n1->GetObject<ConstantPositionMobilityModel> ();
           if(loc1 == 0) {
              loc1 = CreateObject<ConstantPositionMobilityModel> ();
              Vector vec1(save_x - 10,pos_y,0);
              loc1->SetPosition(vec1);
              n1->AggregateObject(loc1);
              //cout << "n1: " <<  pos_x << " " << pos_y << endl;
           }
						save_x = save_x + 20;
            // Place the nodes at suitable position for Animation
           Ptr<Node> n2 = c[subnet_index].Get(1);
           Ptr<ConstantPositionMobilityModel> loc2 = n2->GetObject<ConstantPositionMobilityModel> ();
           if(loc2 == 0) {
              loc2 = CreateObject<ConstantPositionMobilityModel> ();
              Vector vec2(save_x + 10,pos_y ,0);
              loc2->SetPosition(vec2);
              n2->AggregateObject(loc2);
              //cout << "n2: " <<  pos_x << " " << pos_y << endl;
           }
           //printf("Pod placed\n");
	   internet.Install(c[subnet_index]);
           
       p2p.SetDeviceAttribute ("DataRate",DataRateValue (DataRate(SERVER_LINK_CAPACITY)));
       p2p.SetChannelAttribute ("Delay",TimeValue (LINK_LATENCY));
       p2p.SetQueue("ns3::DropTailQueue");
	   // Connect jth subnet with jth router of the POD, there are k/2 PODS.
           for(int q=0;q<k/2;q++){
	   d[temp*2+q] = p2p.Install (NodeContainer (l1[i].Get(j), c[subnet_index].Get(q)));
           bzero(str1,30);
           bzero(str3,30);
           bzero(str2,30);
           bzero(str4,30);
           strcpy(str1,"10.");
           snprintf(str2,10,"%d",i);
           strcat(str2,".");
           snprintf(str3,10,"%d",j);
	   strcat(str3,".");
           snprintf(str4,10,"%d",k*q);
           strcat(str3,str4);
           strcat(str2,str3);
           strcat(str1,str2);
           //cout << "Subnet Address : " << str1 << endl;
           address.SetBase (str1, "255.255.255.252");
           ip[temp*2+q] = address.Assign(d[temp*2+q]);
	   //cout << "Temp : " << temp << "Subnet_index : " << subnet_index << endl;
	   
           }
           subnet_index++;
        }
	//l1_index =(i * k/2 );  
	// Interconnect two rows of the aggregate routers.
	// variale to hold the second position
	int sec_ip = k/2;
	int third_ip = 0;

	p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate (SWITCH_LINK_CAPACITY)));

	for(j=k/2; j < k; j++)
	{
	      //cout << " Combined  Pod " << i << " th " << j << "Router\n"; 
	      for ( p = 0; p < k/2; p++)
	      {  
                 
                 p2p_d[l1_index] = p2p.Install(NodeContainer(l1[i].Get(j),l1[i].Get(p)));
		 //p2p_d[l1_index].Add (csma.Install(NodeContainer ( l1[i].Get (p)))); 
	         //cout << " Router added are  Pod " << i << " th " << p << "Router\n"; 
	         bzero(str1,30);
                 bzero(str3,30);
                 bzero(str2,30);
                 strcpy(str1,"10.");
                 snprintf(str2,10,"%d",i);
                 strcat(str2,".");
		 strcat(str1,str2);
                 snprintf(str3,10,"%d",sec_ip);
		 strcat(str1,str3);
		 strcat(str1,".");
		 bzero(str3,30);
		 snprintf(str3,10,"%d",third_ip);
                 strcat(str1,str3);
                 //cout << "Level: " << i  << " Router: " << j <<" -- IP Address : " << str1 << endl;
                 address.SetBase (str1, "255.255.255.252");
	         ip_for_p2p[l1_index] = address.Assign ( p2p_d[l1_index]);
                 pairx = ip_for_p2p[l1_index].Get (0);
	         //cout << "++++++++++++++l1_index +++++++++++++++++ :" << l1_index << endl;
                 //cout << "Router addresses 1 :" << pairx.first -> GetAddress(1,0) << endl; 
                 //cout << "Router addresses 2 :" << pairx.first -> GetAddress(2,0) << endl; 
                 //cout << "Router addresses 2 :" << pairx.first -> GetAddress(3,0) << endl; 
	         l1_index++;
		 third_ip = third_ip + 4;
		 if( third_ip == 252){
		      third_ip = 0;
		      sec_ip++;
		 }     
	      }
	}
}

// There are (k/2) ^ 2 core routers i.e for k = 8 , 16 core switches
int l2_no = pow ( (k/2), 2);
// These (k/2) ^ 2 core switches are partitioned into sets each containing (k/2) switches
int groups = l2_no / (k/2);
int l2_index = 0;
// Have ip1 as counter for third octet
int ip1 = 1;
// Have ip2 as counter for fourth octet 
int ip2 = 0;

p2p.SetDeviceAttribute ("DataRate", DataRateValue (DataRate (SWITCH_LINK_CAPACITY)));

for( i = 0 ; i < groups ; i++)
{
    for ( j = 0; j < k/2 ; j++)
    {
    	for ( p = 0; p < k ; p++)
    	{
    	    core_d[l2_index] = p2p.Install(NodeContainer(l2.Get((k/2)*i+j),l1[p].Get((k/2) + j)));
	        //cout << "Core****" << k/2*i+j << "****Combined with \n";
        	 //core_d[l2_index].Add (p2p.Install(NodeContainer ( l1[p].Get ((k/2) + j))));
		     //cout << "Pod " << p << " Switch " << (k/2) + j << endl;
    	         bzero(str1,30);
   	         bzero(str3,30);
   	         bzero(str2,30);
    	         strcpy(str1,"10.");
    	         snprintf(str2,10,"%d",k);
		 strcat(str1,str2);
    	         strcat(str1,".");
		 bzero(str2,30);
		 snprintf(str2,10,"%d",ip1);
		 strcat(str1,str2);
		 strcat(str1,".");
                 bzero(str2,30);
		 snprintf(str2,10,"%d",ip2);
		 strcat(str1,str2);
    	         //cout << "Group:  " << i  << " Router: " << j <<" -- IP Address : " << str1 << endl;
    	         address.SetBase (str1, "255.255.255.252");
    	         ip_for_core[l2_index] = address.Assign ( core_d[l2_index]);
	         //uint32_t nNodes = ip_for_core[l2_index].GetN ();
	         //cout << "Number of Nodes at core :" << l2_index << "   is-_-_-_-_ " << nNodes << endl; 
    	         l2_index++;
		 ip2 = ip2 + 4;
		 if(ip2 == 252) {
		    ip2 = 0;
		    ip1++;
		 }   
    	}
    }
}
   
   /********** link utilization ********************************/
  for(int m=0;m<4;m++){
   NetDeviceContainer agg_devs;
  for (NodeContainer::Iterator i = l1[m].Begin (); i != l1[m].End (); ++i)
    {
      Ptr<Node> node = *i;
      //cout<<"device sum:"<<node->GetNDevices ()<<endl;
      for (uint32_t j = 0; j < node->GetNDevices (); ++j)
        {
          agg_devs.Add (node->GetDevice (j));
        }
    }

  // Setup trace on all devices
  for (NetDeviceContainer::Iterator i = agg_devs.Begin (); i != agg_devs.End (); ++i)
    {
      Ptr<NetDevice> dev = *i;
     
      std::ostringstream oss;
        //cout<<"setuptxtrace"<<endl;
        int node_id = dev->GetNode ()->GetId ();
        int dev_id = dev->GetIfIndex();
        oss << "/NodeList/"<<node_id<<"/DeviceList/"<<dev_id<<"/$ns3::PointToPointNetDevice/TxQueue/Dequeue";
        Config::ConnectWithoutContext(oss.str(),MakeBoundCallback(&MOGymEnv::TxTrace,myGymEnv,node_id,dev_id));
    }
  }
  
   NetDeviceContainer core_devs;
   for(NodeContainer::Iterator i = l2.Begin(); i!=l2.End();++i){
       Ptr<Node> node = *i;
      for (uint32_t j = 0; j < node->GetNDevices (); ++j)
        {
          core_devs.Add (node->GetDevice (j));
        }
   }
   
   for (NetDeviceContainer::Iterator i = core_devs.Begin (); i != core_devs.End (); ++i)
    {
      Ptr<NetDevice> dev = *i;
      //MOGymEnv::SetupTxTrace( dev->GetNode ()->GetId (), dev->GetIfIndex ());
       std::ostringstream oss;
        //cout<<"setuptxtrace"<<endl;
        int node_id = dev->GetNode ()->GetId ();
        int dev_id = dev->GetIfIndex();
        oss << "/NodeList/"<<node_id<<"/DeviceList/"<<dev_id<<"/$ns3::PointToPointNetDevice/TxQueue/Dequeue";
        Config::ConnectWithoutContext(oss.str(),MakeBoundCallback(&MOGymEnv::TxTrace,myGymEnv,node_id,dev_id));
     }

    /*********************************************************/
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
   //Print total time taken

  NS_LOG_INFO("Initialize CDF table");
  struct cdf_table* cdfTable = new cdf_table();
  init_cdf(cdfTable);
  load_cdf(cdfTable, cdfFileName.c_str());

  Ipv4GlobalRoutingHelper::RecomputeRoutingTables ();
   
   /* one-to-many */
   /*
   uint32_t nflows = k*k*k/4 - 1;
   uint16_t port_s = 5000;
   uint16_t port_c = 9;
   ApplicationContainer sinkApps;
   
   ApplicationContainer sourceApps;
   
   //source.SetAttribute ("SendSize", UintegerValue (tcp_adu_size));
   
   for(uint32_t i=0;i<nflows;i++)
   {
      PacketSinkHelper sink ("ns3::TcpSocketFactory",InetSocketAddress(ip[0].GetAddress (1), port_s+i));
      sinkApps.Add(sink.Install (c[0].Get (0)));
   }
   sinkApps.Start(Seconds(0.5));
   sinkApps.Stop(Seconds(10.0));
  


   int temp_index=0;
   for(int i=0;i<k*k/2;i++)
   {
      for(int j=0;j<k/2;j++){
          if(i+j==0)
            continue;
          OnOffHelper source ("ns3::TcpSocketFactory",Address(InetSocketAddress (ip[2*i+j].GetAddress (1), port_c)));
          source.SetAttribute ("MaxBytes", UintegerValue (maxBytes));   
          source.SetAttribute ("Remote", AddressValue (InetSocketAddress (ip[0].GetAddress (1), port_s+temp_index)));
          printf( " Pod %d server %d sending data to Pod 0 server 0 \n", i,j);
          source.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=50]"));
          source.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
         
          sourceApps.Add(source.Install(c[i].Get(j)));
          temp_index++;
      }
   }
   sourceApps.Start(Seconds(0.5));
   sourceApps.Stop(Seconds(10.0));
   */

   /* East-west flow */
   long totalFlowSize = 0;

   int temp1;
   uint16_t port = 9;
   ApplicationContainer sourceApps;
   ApplicationContainer sinkApps;

   BulkSendHelper source ("ns3::TcpSocketFactory",Address(InetSocketAddress (ip[0].GetAddress (1), port)));
   //source.SetAttribute ("MaxBytes", UintegerValue (maxBytes));
   //source.SetAttribute ("SendSize", UintegerValue (tcp_adu_size));
   PacketSinkHelper sink ("ns3::TcpSocketFactory",Address(InetSocketAddress (Ipv4Address::GetAny (), port)));

   
   for( int i = 0; i < k ; i++)
   {
       temp = i * k/2;
       for ( int j= 0; j < k; j++){
           if( i != j) {
              temp1 = j * k/2;
              for( int l = 0; l < k/2 ; l++){
                   //printf( " Pod %d server %d sending data to Pod %d server %d \n", temp+l,l,temp1+l,l);
                   source.SetAttribute ("Remote", AddressValue (InetSocketAddress (ip[(temp1+l)*2+l].GetAddress (1), port)));
                   

                   uint32_t flowSize = gen_random_cdf(cdfTable);
                   //uint32_t tos = rand() % 5;

                   totalFlowSize += flowSize;

                   source.SetAttribute ("SendSize", UintegerValue (PACKET_SIZE));
                   source.SetAttribute ("MaxBytes", UintegerValue(flowSize));
                   //source.SetAttribute ("SimpleTOS", UintegerValue (tos));

                   sourceApps.Add(source.Install (c[temp + l].Get (l)));
                   sourceApps.Start (Seconds (0.2));
   		            sourceApps.Stop (Seconds (2.0));
    

                   sink.SetAttribute ("Local",AddressValue(InetSocketAddress(Ipv4Address::GetAny (), port)));
                   sinkApps.Add(sink.Install (c[temp1+l].Get (l)));
                   
                   sinkApps.Start (Seconds (0.2));
   		            sinkApps.Stop (Seconds (2.0));

                  
                   port++;
               }
           }
        }
   }

  
  // Code for flowmonitor
     Ptr<FlowMonitor> flowmon;
     FlowMonitorHelper flowmonHelper;
       //flowmonHelper.SetMonitorAttribute("DelayBinWidth",ns3::DoubleValue(10));
       //flowmonHelper.SetMonitorAttribute("JitterBinWidth",ns3::DoubleValue(10));

       flowmon = flowmonHelper.InstallAll();

       flowmon->CheckForLostPackets();

       std::stringstream flowMonitorFilename;

       Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmonHelper.GetClassifier());

  // End of flow monitor

  //
  // Now, do the actual simulation.
  //
     NS_LOG_INFO ("Run Simulation.");
     Simulator::Stop (Seconds (END_TIME));


     AnimationInterface anim(animFile);
     anim.SetMaxPktsPerTraceFile(99999999999999);
	
     Simulator::Run ();


     std::map< FlowId, FlowMonitor::FlowStats> stats = flowmon->GetFlowStats ();
     int delay_num=0;
     for (std::map<FlowId, FlowMonitor::FlowStats>::iterator flow=stats.begin(); flow!=stats.end();flow++)
     {
          Ipv4FlowClassifier::FiveTuple tuple = classifier->FindFlow(flow->first);

          if(tuple.protocol == 17 && tuple.sourcePort == 698)
            continue;
          //dataset.Add((double)flow->first, (double)flow->second.delaySum.GetSeconds() / (double)flow->second.rxPackets);
          std::cout<<flow->first<<endl;
          print_stats ( flow->second);
          cout<<"fct:"<<(double)flow->second.timeLastRxPacket.GetSeconds() - (double) flow->second.timeFirstTxPacket.GetSeconds();

          if(flow->second.rxBytes <27000 &&((double)flow->second.timeLastRxPacket.GetSeconds() - (double) flow->second.timeFirstTxPacket.GetSeconds())>0.002){
            delay_num++;
          }
           cout<<"delay_num:"<<delay_num<<endl;

          //dataset.Add((double)flow->first, (double)flow->second.timeLastRxPacket.GetSeconds() - (double) flow->second.timeFirstTxPacket.GetSeconds());
     }
      myGymEnv->m_fct=delay_num;



     myGymEnv->NotifySimulationEnd();
     Simulator::Destroy ();
     free_cdf(cdfTable);
     cout<< "sinkapps"<<sinkApps.GetN()<<endl;
     
     for(uint32_t i=0;i<sinkApps.GetN();i++){
     Ptr<PacketSink> sink1 = DynamicCast<PacketSink> (sinkApps.Get(i));
     cout << "Total Bytes received: " << sink1->GetTotalRx () << endl;
     }

    //  anim.StopAnimation ();

     NS_LOG_INFO ("Done.");
     NS_LOG_UNCOND("Done");

}

