#ifndef MO_ENV_H
#define MO_ENV_H

#include "ns3/opengym-module.h"
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <sstream>
#include <map>
#include <utility>
#include <set>

#include <stdio.h>
#include <stdlib.h>

#include "ns3/object.h"
#include "ns3/node-list.h"
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



namespace ns3 {

class Time;


class MOGymEnv : public OpenGymEnv
{
public:
  MOGymEnv (Time stepTime);
  virtual ~MOGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  static void TxTrace(Ptr<MOGymEnv> entity, int dev_id, int ith_id,Ptr<Packet const> packet);

  //std::string GetTcpCongStateName(const TcpSocketState::TcpCongState_t state);
  //std::string GetTcpCAEventName(const TcpSocketState::TcpCAEvent_t event);

  // OpenGym interface
  Ptr<OpenGymSpace> GetActionSpace();
  bool GetGameOver();
  float GetReward();
  std::string GetExtraInfo();
  bool ExecuteActions(Ptr<OpenGymDataContainer> action);
  Ptr<OpenGymSpace> GetObservationSpace();
  Ptr<OpenGymDataContainer> GetObservation();
  int m_fct = 0;

  void setStartTime(int node_id, int dev_id,double time);
  void setEndTime(int node_id, int dev_id,double time);
  void setTotalBytes(int node_id, int dev_id,uint32_t pksize);


protected:
  uint32_t m_obs_link_num;
  uint32_t m_op_link_num;
  uint32_t m_node_num;
  uint32_t m_pod_num;

  // state
  // obs has to be implemented in child class

  // game over
  bool m_isGameOver;


  // extra info
  std::string m_info;


private:
  // reward
  float m_reward1=0;
  int m_reward0=0;
  

  void ScheduleNextStateRead();

  Time m_duration;
  Time m_timeStep;
  Time m_interval = Seconds(0.005);

  std::vector < std::vector<uint64_t>> m_total_bytes = std::vector<std::vector<uint64_t>>(40,std::vector<uint64_t>(8,0));
  std::vector < std::vector<double>> m_start_time = std::vector < std::vector<double>>(40,std::vector<double>(8,0));
  std::vector < std::vector<double>> m_end_time = std::vector < std::vector<double>>(40, std::vector<double>(8,0));
  std::vector < double > m_link_utilization = std::vector < double >(40,0);

  // (A,B) : node A device(interface) B
  std::vector<std::pair<int,int>> m_op_link_interface_first = {std::make_pair(7,1),std::make_pair(7,2),std::make_pair(15,1),std::make_pair(15,2),
                                                       std::make_pair(23,1),std::make_pair(23,2),std::make_pair(31,1),std::make_pair(31,2),
                                                       std::make_pair(1,1),std::make_pair(1,2),std::make_pair(1,3),std::make_pair(1,4),
                                                       std::make_pair(2,1),std::make_pair(2,2),std::make_pair(2,3),std::make_pair(2,4),
                                                       std::make_pair(3,1),std::make_pair(3,2),std::make_pair(3,3),std::make_pair(3,4)};
  std::vector<std::pair<int,int>> m_op_link_interface_second = {std::make_pair(4,4),std::make_pair(5,4),std::make_pair(12,4),std::make_pair(13,4),
                                                       std::make_pair(20,4),std::make_pair(21,4),std::make_pair(28,4),std::make_pair(29,4),
                                                       std::make_pair(7,3),std::make_pair(15,3),std::make_pair(23,3),std::make_pair(31,3),
                                                       std::make_pair(6,4),std::make_pair(14,4),std::make_pair(22,4),std::make_pair(30,4),
                                                       std::make_pair(7,4),std::make_pair(15,4),std::make_pair(23,4),std::make_pair(31,4)};

  std::vector<std::pair<int,int>> m_obs_link_interface_first = {std::make_pair(6,1),std::make_pair(6,2),std::make_pair(7,1),std::make_pair(7,2),
                                                                std::make_pair(14,1),std::make_pair(14,2),std::make_pair(15,1),std::make_pair(15,2),
                                                           std::make_pair(22,1),std::make_pair(22,2),std::make_pair(23,1),std::make_pair(23,2),
                                                           std::make_pair(30,1),std::make_pair(30,2),std::make_pair(31,1),std::make_pair(31,2),
                                                           std::make_pair(0,1),std::make_pair(0,2),std::make_pair(0,3),std::make_pair(0,4),
                                                           std::make_pair(1,1),std::make_pair(1,2),std::make_pair(1,3),std::make_pair(1,4),
                                                           std::make_pair(2,1),std::make_pair(2,2),std::make_pair(2,3),std::make_pair(2,4),
                                                           std::make_pair(3,1),std::make_pair(3,2),std::make_pair(3,3),std::make_pair(3,4)};
  std::vector<std::pair<int,int>> m_obs_link_interface_second ={std::make_pair(4,3),std::make_pair(5,3),std::make_pair(4,4),std::make_pair(5,4),
                                                               std::make_pair(12,3),std::make_pair(13,3),std::make_pair(12,4),std::make_pair(13,4),
                                                          std::make_pair(20,3),std::make_pair(21,3),std::make_pair(20,4),std::make_pair(21,4),
                                                          std::make_pair(28,3),std::make_pair(29,3),std::make_pair(28,4),std::make_pair(29,4),
                                                          std::make_pair(6,3),std::make_pair(14,3),std::make_pair(22,3),std::make_pair(30,3),
                                                          std::make_pair(7,3),std::make_pair(15,3),std::make_pair(23,3),std::make_pair(31,3),
                                                          std::make_pair(6,4),std::make_pair(14,4),std::make_pair(22,4),std::make_pair(30,4),
                                                          std::make_pair(7,4),std::make_pair(15,4),std::make_pair(23,4),std::make_pair(31,4)};
};

} // namespace ns3

#endif /* MO_ENV_H */
