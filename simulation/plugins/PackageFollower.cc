#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <array>
#include <mutex>
#include <vector>

namespace mupt {
class PackageFollower : public gz::sim::System, public gz::sim::ISystemConfigure,
                        public gz::sim::ISystemPreUpdate {
  struct State {unsigned carrier=0; bool dirty=false; gz::math::Pose3d drop;};
  std::array<State,6> state;
  std::mutex mutex;
  gz::transport::Node node;
  bool OnState(const gz::msgs::Pose_V &request,gz::msgs::Boolean &reply) {
    std::vector<std::pair<unsigned,State>> updates;
    for (const auto &p:request.pose()) {
      unsigned number=0;
      try {
        if (p.name().rfind("package_",0)!=0) throw std::invalid_argument("name");
        number=std::stoul(p.name().substr(8));
      } catch (...) {reply.set_data(false);return true;}
      if (number<1 || number>6 || p.id()>3) {reply.set_data(false);return true;}
      State s;s.carrier=p.id();s.dirty=true;
      s.drop=gz::math::Pose3d(p.position().x(),p.position().y(),p.position().z(),0,0,0);
      updates.push_back({number-1,s});
    }
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &u:updates) state[u.first]=u.second;
    reply.set_data(true);return true;
  }
  public: void Configure(const gz::sim::Entity &world,const std::shared_ptr<const sdf::Element> &,
      gz::sim::EntityComponentManager &ecm,gz::sim::EventManager &) override {
    auto name=ecm.Component<gz::sim::components::Name>(world)->Data();
    node.Advertise("/world/"+name+"/package_state",&PackageFollower::OnState,this);
  }
  public: void PreUpdate(const gz::sim::UpdateInfo &info,gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    std::lock_guard<std::mutex> lock(mutex);
    for (unsigned i=0;i<state.size();++i) {
      auto &s=state[i];if (!s.carrier && !s.dirty) continue;
      auto package=ecm.EntityByComponents(gz::sim::components::Model(),gz::sim::components::Name("package_"+std::to_string(i+1)));
      if (package==gz::sim::kNullEntity) continue;
      gz::math::Pose3d pose=s.drop;
      if (s.carrier) {
        auto vehicle=ecm.EntityByComponents(gz::sim::components::Model(),gz::sim::components::Name("x500_"+std::to_string(s.carrier)));
        if (vehicle==gz::sim::kNullEntity) continue;
        unsigned count=0,slot=0;
        for (unsigned j=0;j<state.size();++j) if (state[j].carrier==s.carrier) {++count;if (j<i) ++slot;}
        pose=gz::sim::worldPose(vehicle,ecm);
        pose.Pos().X()+=(static_cast<double>(slot)-(count-1)/2.)*.45;
        pose.Pos().Z()-=.65;
        pose.Rot()=gz::math::Quaterniond::Identity;
      }
      gz::sim::Model(package).SetWorldPoseCmd(ecm,pose);
      ecm.SetComponentData<gz::sim::components::Pose>(package,pose);
      s.dirty=false;
    }
  }
};
}
GZ_ADD_PLUGIN(mupt::PackageFollower,gz::sim::System,gz::sim::ISystemConfigure,gz::sim::ISystemPreUpdate)
