#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/math/Vector3.hh>
#include <sdf/Element.hh>
#include <chrono>
#include <cmath>
#include <vector>

namespace mupt {
class CityObstacles : public gz::sim::System, public gz::sim::ISystemConfigure,
                      public gz::sim::ISystemPreUpdate {
  struct Orbit {std::string name; gz::math::Vector3d c,a,w,p;};
  std::vector<Orbit> orbits;
  public: void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    auto config = sdf->Clone();
    if (!config->HasElement("obstacle")) return;
    for (auto e=config->GetElement("obstacle"); e; e=e->GetNextElement("obstacle"))
      orbits.push_back({e->Get<std::string>("name"), e->Get<gz::math::Vector3d>("center"),
        e->Get<gz::math::Vector3d>("amplitude"), e->Get<gz::math::Vector3d>("rate"),
        e->Get<gz::math::Vector3d>("phase")});
  }
  public: void PreUpdate(const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    double t=std::chrono::duration<double>(info.simTime).count();
    for (const auto &o: orbits) {
      auto entity=ecm.EntityByComponents(gz::sim::components::Model(), gz::sim::components::Name(o.name));
      if (entity==gz::sim::kNullEntity) continue;
      gz::math::Pose3d pose(o.c.X()+o.a.X()*std::cos(o.w.X()*t+o.p.X()),
        o.c.Y()+o.a.Y()*std::sin(o.w.Y()*t+o.p.Y()),
        o.c.Z()+o.a.Z()*std::sin(o.w.Z()*t+o.p.Z()),0,0,0);
      gz::sim::Model(entity).SetWorldPoseCmd(ecm,pose);
      ecm.SetComponentData<gz::sim::components::Pose>(entity,pose);
    }
  }
};
}
GZ_ADD_PLUGIN(mupt::CityObstacles, gz::sim::System, gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate)
