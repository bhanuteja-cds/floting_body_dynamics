#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    #if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
    #  define USE_PETSC_LA
    #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
    #else
    #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
    #endif
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/sparse_ilu.h>

#include <iostream>
#include <fstream>
#include <sstream>

namespace Navierstokes
{
    using namespace dealii;
    
    template <int dim>
    class BoundaryValues105;
    
    template <int dim>
    class moveBoundaryValues105;
    
    template <int dim>
    class RightHandSide;
    
    template <int dim>
    class StokesProblem
    {        
    private:
        MPI_Comm                                  mpi_communicator;
        double deltat = 0.0025;
        double totaltime = 4;
        double diffusivity = 0.0;
        double viscosity = 0.0025, density = 250.0;
        double viscosity_air = 0.0001, density_air = 1;
        double viscosity1 = 0.0, density1 = 0.0;
        double mass_rigidbody = 0.0;
        double density_rbody = 150;
        double rotation_vel_rbody = 0;
        double local_moment;
        double total_moment;
        double moment_of_inertia = 0.0;
        double local_hmin = 100.0, global_hmin;
        int meshrefinement = 0;
        double temp = 1e-6;
        int degree;
        parallel::distributed::Triangulation<dim> triangulation;
        LA::MPI::SparseMatrix                     ns_system_matrix;
        LA::MPI::SparseMatrix                     vof_system_matrix, move_system_matrix;
        DoFHandler<dim>                           dof_handler, vof_dof_handler, move_dof_handler;
        FESystem<dim>                             fe, fevof, femove;
        LA::MPI::Vector                           lr_solution, lr_old_iterate_solution, lr_nonlinear_residue, lr_vof_solution;
        LA::MPI::Vector                           lo_system_rhs, lo_vof_system_rhs, lo_initial_condition_vof, lo_lhs_productvec;
        LA::MPI::Vector                           lr_move_solution, lr_total_meshmove; //lr_old_move_solution
        LA::MPI::Vector                           lo_move_system_rhs, lo_total_meshmove;
        AffineConstraints<double>                 vofconstraints, stokesconstraints, moveconstraints;
        IndexSet                                  owned_partitioning_stokes, owned_partitioning_vof, owned_partitioning_movemesh;
        IndexSet                                  relevant_partitioning_stokes, relevant_partitioning_vof, relevant_partitioning_movemesh;
        ConditionalOStream                        pcout;
        Point<dim>                                centre_of_mass;
        Tensor<1, dim>                            local_force;
        Tensor<1, dim>                            total_force;
        Tensor<1, dim>                            prev_vel_rbody, vel_rbody;
        TimerOutput                               computing_timer;
        
    public:
        void setup_stokessystem();
        void resetup_stokessystem(const BoundaryValues105<dim>);
        void setup_vofsystem();
        void setup_movemeshsystem();
        void resetup_stokessystem();
        void resetup_vofsystem();
        void resetup_movemeshsystem(const moveBoundaryValues105<dim>);
        void assemble_stokessystem(const RightHandSide<dim>);
        void assemble_stokessystem_nonlinear();
        void assemble_vofsystem();
        void assemble_movemesh();
        void solve_stokes();
        void solve_vof();
        void solve_movemesh();
        void movemesh();
        void compute_hmin();
        void compute_stresses();
        void compute_rigidbody_vel();
        double compute_linf_vof();
        double compute_maxvel();
        double compute_nonlinear_residue();
        void compute_meanpressure();
        void compute_masses();
        void output_results (int);
        void timeloop();
        
        StokesProblem(int degreein)
        :
        mpi_communicator (MPI_COMM_WORLD),
        degree(degreein),
        triangulation (mpi_communicator),
        dof_handler(triangulation),
        vof_dof_handler(triangulation),
        move_dof_handler(triangulation),
        fe(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1),
        fevof(FE_Q<dim>(degree), 1),
        femove(FE_Q<dim>(degree), dim),
        pcout (std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        centre_of_mass(0, 0),
        computing_timer (mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
        {      
            pcout << "stokes constructor success...."<< std::endl;
        }
    };
    //=====================================================  
    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide () : Function<dim>(dim+1)
        {}
        virtual void vector_value(const Point<dim> &, Vector<double> &value) const;
    };
    
    template <int dim>
    void RightHandSide<dim>::vector_value(const Point<dim> &,  Vector<double> &values) const
    {
        const double time = this->get_time();
        values[0] = 0.25*sin(2*M_PI*time); //test (+X). equavalent to tow/propeller on ship
//         values[0] = 0;
        values[1] = -10.0; //gravity (-Y)
        values[2] = 0.0;
    }
    //==================================================
    template<int dim>
    class BoundaryValues105 : public Function<dim>
    {
    public:
        BoundaryValues105():Function<dim>(dim+1)
        {}
        virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
        void set_rbody_vels (Tensor<1, dim> &translation_rbody, double rotation_rbody);
        void set_centre_of_mass (Point<dim> &centreofmass);
        Tensor<1, dim> translation_vel;
        double rotation_vel;
        Point<dim> centre_of_mass;
    };
    
    template<int dim>
    void BoundaryValues105<dim>::set_rbody_vels (Tensor<1, dim> &translation_rbody, double rotation_rbody)
    {
        translation_vel = translation_rbody;
        rotation_vel = rotation_rbody;
    }
    
    template<int dim>
    void BoundaryValues105<dim>::set_centre_of_mass (Point<dim> &centreofmass)
    {
        centre_of_mass = centreofmass;
    }
    
    template<int dim>
    void BoundaryValues105<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
            values[0] = 0 - rotation_vel*(p[1] - centre_of_mass[1]);  // no X translation
            values[1] = translation_vel[1] + rotation_vel*(p[0] - centre_of_mass[0]);
            values[2] = 0;
    }
   //==================================================
    template<int dim>
    class moveBoundaryValues105 : public Function<dim>
    {
    public:
        moveBoundaryValues105():Function<dim>(dim)
        {}
        virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
        void set_rbody_vels (Tensor<1, dim> &translation_rbody, double rotation_rbody);
        void set_centre_of_mass (Point<dim> &centreofmass);
        Tensor<1, dim> translation_vel;
        double rotation_vel;
        Point<dim> centre_of_mass;
    };
    
    template<int dim>
    void moveBoundaryValues105<dim>::set_rbody_vels (Tensor<1, dim> &translation_rbody, double rotation_rbody)
    {
        translation_vel = translation_rbody;
        rotation_vel = rotation_rbody;
    }
    
    template<int dim>
    void moveBoundaryValues105<dim>::set_centre_of_mass (Point<dim> &centreofmass)
    {
        centre_of_mass = centreofmass;
    }
    
    template<int dim>
    void moveBoundaryValues105<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
            values[0] = 0 - rotation_vel*(p[1] - centre_of_mass[1]); // no X translation
            values[1] = translation_vel[1] + rotation_vel*(p[0] - centre_of_mass[0]);
    }
    //==================================================  
    template <int dim>
    class TractionBoundaryValues : public Function<dim>
    {
    public:
        TractionBoundaryValues () : Function<dim>(dim)
        {}
        virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &values) const;
    };
    
    template <int dim>
    void  TractionBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &values) const
    {
        //       const double time = this->get_time();
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p][0] = 0.0;
            values[p][1] = 0.0;
        }    
    }
        //==========================================================
    template <int dim>
    class MoveRightHandSide : public TensorFunction<1,dim>
    {
    public:
        MoveRightHandSide() : TensorFunction<1,dim>() {}
        
        virtual void value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1,dim>> &values) const;
    };
    
    template <int dim>
    void  MoveRightHandSide<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<1,dim>> &values) const
    {
        Assert (points.size() == values.size(), ExcDimensionMismatch (points.size(), values.size()));
        
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p].clear ();
            values[p][0] = 0.;
            values[p][1] = 0.;
//             values[p][2] = 0.;
        }
    }
    //===========================================================
//     template <int dim>
//     class VofBoundaryValues : public Function<dim>
//     {
//     public:
//         VofBoundaryValues() : Function<dim>(1) {}
//         virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
//         {
//             if (p[1] < 0.125)
//                 return 1;
//             else
//                 return 0;
//         }
//     };
    //====================================
    template <int dim>
    class VofRightHandSide : public Function<dim>
    {
    public:
        VofRightHandSide() : Function<dim>(1) {}
        virtual double value(const Point<dim> &, const unsigned int component = 0) const override;
    };
    template <int dim>
    double VofRightHandSide<dim>::value(const Point<dim> &, const unsigned int) const
    {
        return 0.0;
    }
    //==========================================
    template <int dim>
    class InitialValues : public Function<dim>
    {
    public:
        int fac = 10;
        double ymax = 0.2/fac;
        double xmax = 1.0/fac;
        InitialValues () : Function<dim>(1) {}
        virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
    };
    template <int dim>
    double InitialValues<dim>::value (const Point<dim> &p, const unsigned int) const
    {        
        if(p[1] < ymax/2+ymax/10.85+sin(4*fac*M_PI*(p[0]))*(ymax/6.5) && !(p[1] > ymax/2 && p[0] < xmax*3/4 && p[0] > xmax/2))
            return 1.0;
        else
            return 0.0;
    }
    //==============================================================  
    template <int dim>
    void StokesProblem<dim>::setup_stokessystem()
    {  
        TimerOutput::Scope t(computing_timer, "setup_stokessystem");
        pcout <<"setup_stokessystem "<<std::endl;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("domain_2D_hex.msh");
        grid_in.read_msh(input_file);
        triangulation.refine_global (meshrefinement);
        dof_handler.distribute_dofs(fe);
        
        
         // computing quantities for the rigid body
        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            GridIn<dim> grid_in_rbody;
            Triangulation<dim> triangulation_rbody;
            FE_Q<dim>          fe_rbody(1);
            DoFHandler<dim>    dof_handler_rbody(triangulation_rbody);
            grid_in_rbody.attach_triangulation(triangulation_rbody);
            std::ifstream input_file("floating_rbody_hex.msh");
            grid_in_rbody.read_msh(input_file);
            triangulation_rbody.refine_global (2);
            dof_handler_rbody.distribute_dofs(fe_rbody);
            
            std::vector<bool> vertex_touched(triangulation_rbody.n_vertices(), false);
            for (typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler_rbody.begin_active(); cell != dof_handler_rbody.end(); ++cell)
            {
                if(cell->is_locally_owned())
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                        if (vertex_touched[cell->vertex_index(v)] == false)
                        {
                            vertex_touched[cell->vertex_index(v)] = true;
                            centre_of_mass += cell->vertex(v);
                        }
            } // end of cell loop
            for(unsigned int d=0; d<dim; ++d)
                centre_of_mass[d] /= vertex_touched.size();
            pcout << "x is " << centre_of_mass[0] << " y is " << centre_of_mass[1] << std::endl;

            QGauss<dim>   quadrature_formula(3);
            FEValues<dim> fe_values (fe_rbody, quadrature_formula,
                                 update_quadrature_points  |
                                 update_JxW_values);
            const unsigned int   n_q_points      = quadrature_formula.size();
            std::vector<Point<dim>> quadrature_points(n_q_points);
            
             for (typename DoFHandler<dim>::active_cell_iterator  cell = dof_handler_rbody.begin_active(); cell != dof_handler_rbody.end(); ++cell)
            {
                fe_values.reinit (cell);
                quadrature_points = fe_values.get_quadrature_points();
                
                for(unsigned int q=0; q<n_q_points; q++)
                {
                moment_of_inertia +=  (pow((quadrature_points[q][0] - centre_of_mass[0]), 2) + pow((quadrature_points[q][1] - centre_of_mass[1]), 2))*fe_values.JxW(q);
                
                mass_rigidbody += fe_values.JxW(q);
                }
            } //end of cell loop for computing moment of inertia
            moment_of_inertia *= density_rbody;
            mass_rigidbody *= density_rbody;
            pcout << "mass of rigid body = " << mass_rigidbody << std::endl;
            pcout << "MOI of rigid body = " << moment_of_inertia << std::endl;
        }
           
           MPI_Bcast(&centre_of_mass[0], 1, MPI_DOUBLE, 0, mpi_communicator);
           MPI_Bcast(&centre_of_mass[1], 1, MPI_DOUBLE, 0, mpi_communicator);
           MPI_Bcast(&mass_rigidbody, 1, MPI_DOUBLE, 0, mpi_communicator);
           MPI_Bcast(&moment_of_inertia, 1, MPI_DOUBLE, 0, mpi_communicator);

        pcout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Total number of cells: "
        << triangulation.n_cells()
        << std::endl;
        pcout << "   Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
        
        std::vector<unsigned int> block_component(dim+1,0);
        block_component[dim] = 1;
        std::vector<types::global_dof_index> dofs_per_block (2);
//         DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
//         DoFTools::count_dofs_per_fe_block (dof_handler, dofs_per_block);
//         const unsigned int n_u = dofs_per_block[0],  n_p = dofs_per_block[1];
//         pcout << " (" << n_u << '+' << n_p << ')' << std::endl;
//         pcout << "dofspercell "<< fe.dofs_per_cell << std::endl;
        
        owned_partitioning_stokes = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (dof_handler, relevant_partitioning_stokes);
        
        {
            stokesconstraints.clear();
            stokesconstraints.reinit(relevant_partitioning_stokes);
            DoFTools::make_hanging_node_constraints (dof_handler, stokesconstraints);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (103);
            no_normal_flux_boundaries.insert (102);
            no_normal_flux_boundaries.insert (104);
//             std::set<types::boundary_id> no_tangential_flux_boundaries;
//             no_tangential_flux_boundaries.insert (102);
//             no_tangential_flux_boundaries.insert (104);
            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim); 
            ComponentMask velocities_mask = fe.component_mask(velocities);
            ComponentMask pressure_mask = fe.component_mask(pressure);
            VectorTools::compute_no_normal_flux_constraints (dof_handler, 0, no_normal_flux_boundaries, stokesconstraints);
//             VectorTools::compute_normal_flux_constraints (dof_handler, 0, no_tangential_flux_boundaries, stokesconstraints);
            VectorTools::interpolate_boundary_values (dof_handler, 105, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
//             VectorTools::interpolate_boundary_values (dof_handler, 102, ZeroFunction<dim>(dim+1), stokesconstraints, velocities_mask);
            stokesconstraints.close();
        }

        ns_system_matrix.clear();        
        DynamicSparsityPattern dsp (relevant_partitioning_stokes);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, stokesconstraints, false);
        SparsityTools::distribute_sparsity_pattern (dsp, dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_stokes);
        ns_system_matrix.reinit (owned_partitioning_stokes, owned_partitioning_stokes, dsp, mpi_communicator);        
        lr_solution.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lr_old_iterate_solution.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lr_nonlinear_residue.reinit(owned_partitioning_stokes, relevant_partitioning_stokes, mpi_communicator);
        lo_system_rhs.reinit(owned_partitioning_stokes, mpi_communicator);
        lo_lhs_productvec.reinit(owned_partitioning_stokes, mpi_communicator);
    }
    //==============================================================  
    template <int dim>
    void StokesProblem<dim>::resetup_stokessystem(const BoundaryValues105<dim> boundaryvalues105)
    {
        {
            pcout << "resetup_stokessystem "<< std::endl;
            stokesconstraints.clear();
            stokesconstraints.reinit(relevant_partitioning_stokes);
            DoFTools::make_hanging_node_constraints (dof_handler, stokesconstraints);
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert (101);
            no_normal_flux_boundaries.insert (102);
            no_normal_flux_boundaries.insert (103);
            no_normal_flux_boundaries.insert (104);
//             std::set<types::boundary_id> no_tangential_flux_boundaries;
//             no_tangential_flux_boundaries.insert (102);
//             no_tangential_flux_boundaries.insert (104);
            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);        
            ComponentMask velocities_mask = fe.component_mask(velocities);
            ComponentMask pressure_mask = fe.component_mask(pressure);
            VectorTools::compute_no_normal_flux_constraints (dof_handler, 0, no_normal_flux_boundaries, stokesconstraints);
//             VectorTools::compute_normal_flux_constraints (dof_handler, 0, no_tangential_flux_boundaries, stokesconstraints);
            VectorTools::interpolate_boundary_values (dof_handler, 105, boundaryvalues105, stokesconstraints, velocities_mask);
            stokesconstraints.close();
        }
    }  
    //========================================================  
    template <int dim>
    void StokesProblem<dim>::setup_vofsystem()
    { 
        TimerOutput::Scope t(computing_timer, "setup_vofsystem");
        pcout <<"setup_vofsystem "<<std::endl;
        vof_dof_handler.distribute_dofs(fevof);
        owned_partitioning_vof = vof_dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (vof_dof_handler, relevant_partitioning_vof);
        
        {
            vofconstraints.clear();
            vofconstraints.reinit(relevant_partitioning_vof);
            //             DoFTools::make_hanging_node_constraints (vof_dof_handler, vofconstraints);
            //             std::set<types::boundary_id> no_normal_flux_boundaries;
            //             no_normal_flux_boundaries.insert (101);
            //             no_normal_flux_boundaries.insert (102);
            //             VectorTools::compute_no_normal_flux_constraints (move_dof_handler, 0, no_normal_flux_boundaries, moveconstraints);
            //             VectorTools::interpolate_boundary_values (move_dof_handler, 104, ZeroFunction<dim>(dim), moveconstraints);
            vofconstraints.close();
        }
        pcout << "Number of vof degrees of freedom: " << vof_dof_handler.n_dofs() << std::endl; 
        vof_system_matrix.clear();
        DynamicSparsityPattern vof_dsp(relevant_partitioning_vof);
        DoFTools::make_sparsity_pattern(vof_dof_handler, vof_dsp, vofconstraints, false);
        SparsityTools::distribute_sparsity_pattern (vof_dsp, vof_dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_vof);        
        vof_system_matrix.reinit(owned_partitioning_vof, owned_partitioning_vof, vof_dsp, mpi_communicator);
        
        lr_vof_solution.reinit(owned_partitioning_vof, relevant_partitioning_vof, mpi_communicator);
        lo_vof_system_rhs.reinit(owned_partitioning_vof, mpi_communicator);
        lo_initial_condition_vof.reinit(owned_partitioning_vof, mpi_communicator);
        
        InitialValues<dim> initialcondition;
        VectorTools::interpolate(vof_dof_handler, initialcondition, lo_initial_condition_vof);
        lr_vof_solution = lo_initial_condition_vof;
    }
    //==============================================================    
    template <int dim>
    void StokesProblem<dim>::setup_movemeshsystem()
    { 
        TimerOutput::Scope t(computing_timer, "setup_movemeshsystem");
        pcout <<"setup_movemeshsystem "<<std::endl;
        move_dof_handler.distribute_dofs(femove);
        owned_partitioning_movemesh = move_dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (move_dof_handler, relevant_partitioning_movemesh);
        moveBoundaryValues105<dim> moveboundaryvalues105;

        {
            moveconstraints.reinit(relevant_partitioning_movemesh);
            DoFTools::make_hanging_node_constraints (move_dof_handler, moveconstraints);
            std::set<types::boundary_id> no_normal_flux_boundaries;
//             no_normal_flux_boundaries.insert (101);
//             VectorTools::compute_no_normal_flux_constraints (move_dof_handler, 0, no_normal_flux_boundaries, moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 101, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 102, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 103, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 104, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 105, moveboundaryvalues105, moveconstraints);
            moveconstraints.close();
        }
        pcout << "Number of move degrees of freedom: " << move_dof_handler.n_dofs() << std::endl; 
        move_system_matrix.clear();
        DynamicSparsityPattern move_dsp(relevant_partitioning_movemesh);
        DoFTools::make_sparsity_pattern(move_dof_handler, move_dsp, moveconstraints, false);
        SparsityTools::distribute_sparsity_pattern (move_dsp, move_dof_handler.locally_owned_dofs(), mpi_communicator, relevant_partitioning_movemesh);
        move_system_matrix.reinit(owned_partitioning_movemesh, owned_partitioning_movemesh, move_dsp, mpi_communicator);
        lr_move_solution.reinit(owned_partitioning_movemesh, relevant_partitioning_movemesh, mpi_communicator);
//         lr_old_move_solution.reinit(owned_partitioning_movemesh, relevant_partitioning_movemesh, mpi_communicator);
        lr_total_meshmove.reinit(owned_partitioning_movemesh, relevant_partitioning_movemesh, mpi_communicator);
        lo_total_meshmove.reinit(owned_partitioning_movemesh, mpi_communicator);
        lo_move_system_rhs.reinit(owned_partitioning_movemesh, mpi_communicator);
    } 
    //===========================================================
    template <int dim>
    void StokesProblem<dim>::resetup_movemeshsystem(const moveBoundaryValues105<dim> moveboundaryvalues105)
    { 
        pcout <<"resetup_movemeshsystem "<<std::endl;            
        {
            moveconstraints.reinit(relevant_partitioning_movemesh);
            std::set<types::boundary_id> no_normal_flux_boundaries;
//             no_normal_flux_boundaries.insert (101);
//             VectorTools::compute_no_normal_flux_constraints (move_dof_handler, 0, no_normal_flux_boundaries, moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 101, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 102, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 103, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 104, ZeroFunction<dim>(dim), moveconstraints);
            VectorTools::interpolate_boundary_values (move_dof_handler, 105, moveboundaryvalues105, moveconstraints);
            moveconstraints.close();
        }
    }
    //===========================================================
    template <int dim>
    void StokesProblem<dim>::assemble_stokessystem(const RightHandSide<dim> righthandside)
    {
        TimerOutput::Scope t(computing_timer, "assemble_stokessystem");
        pcout <<"assemble_stokessystem "<<std::endl;
        ns_system_matrix=0;
        lo_system_rhs=0;
        
        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        FEValues<dim> fe_move_values (femove, quadrature_formula, update_values | update_quadrature_points);
        
        FEValues<dim> fe_vof_values (fevof, quadrature_formula,
                                     update_values    |
                                     update_gradients |
                                     update_quadrature_points  |
                                     update_JxW_values);
        
        
//         FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
//                                           update_values    | update_normal_vectors |
//                                           update_quadrature_points  | update_JxW_values);
        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
//         const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim); 
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index>  local_dof_indices (dofs_per_cell);        
//         const RightHandSide<dim>              right_hand_side;
//         const TractionBoundaryValues<dim>     traction_boundary_values;
        std::vector<Vector<double>>           rhs_values(n_q_points, Vector<double>(dim+1));
//         std::vector<Vector<double>>           neumann_boundary_values(n_face_q_points, Vector<double>(dim+1));        
        std::vector<Tensor<1, dim>>           value_phi_u (dofs_per_cell);    
        std::vector<Tensor<2, dim>>           gradient_phi_u (dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>>  symgrad_phi_u (dofs_per_cell);
        std::vector<double>                   div_phi_u(dofs_per_cell);
        std::vector<double>                   phi_p(dofs_per_cell);        
        std::vector<Tensor<2, dim> >          old_velocity_gradients(n_q_points);
        std::vector<Tensor<1, dim> >          old_solution_values(n_q_points);
        std::vector<double>                   vof_values(n_q_points);
//         std::vector<Tensor<1, dim> >          vof_values_gradients(n_q_points);
        std::vector<Tensor<1, dim> >          meshvelocity_values(n_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator vof_cell = vof_dof_handler.begin_active();
        typename DoFHandler<dim>::active_cell_iterator cell_move = move_dof_handler.begin_active();
        
        for (; cell!=endc; ++cell)
        { 
            if (cell->is_locally_owned())
            {
                fe_values.reinit (cell);
                fe_vof_values.reinit(vof_cell);
                fe_move_values.reinit(cell_move);
                fe_values[velocities].get_function_values(lr_old_iterate_solution, old_solution_values);
//                 fe_values[velocities].get_function_gradients(lr_old_iterate_solution, old_velocity_gradients);
                fe_vof_values.get_function_values(lr_vof_solution, vof_values);
                fe_move_values[velocities].get_function_values(lr_move_solution, meshvelocity_values);
                
                local_matrix = 0;
                local_rhs = 0;
//                 right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
                righthandside.vector_value_list(fe_values.get_quadrature_points(), rhs_values);

                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    viscosity1 = vof_values[q]*viscosity +(1-vof_values[q])*viscosity_air;
//                     viscosity1 *= 4;
                    density1   = vof_values[q]*density +(1-vof_values[q])*density_air;                     
                    
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {                        
                        value_phi_u[k]   = fe_values[velocities].value (k, q);
                        gradient_phi_u[k]= fe_values[velocities].gradient (k, q);
                        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                        div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                        phi_p[k]         = fe_values[pressure].value (k, q);
                    }
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {                    
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {                            
                            local_matrix(i, j) += (density1*value_phi_u[j]*value_phi_u[i] + 
                            deltat*density1*(gradient_phi_u[j] * (old_solution_values[q] - meshvelocity_values[q]))*value_phi_u[i] +
//                             deltat*density1*(gradient_phi_u[j] * (old_solution_values[q]))*value_phi_u[i] +
                            2*deltat*viscosity1*scalar_product(symgrad_phi_u[j], symgrad_phi_u[i]) -
                            deltat * phi_p[j] * div_phi_u[i] - 
                            div_phi_u[j] * phi_p[i]) * fe_values.JxW(q); //implicit
                            
//                             local_matrix(i, j) += (density1*value_phi_u[j]*value_phi_u[i] +
//                             2*deltat*viscosity1*scalar_product(symgrad_phi_u[j], symgrad_phi_u[i]) -
//                             deltat * phi_p[j] * div_phi_u[i] - 
//                             div_phi_u[j] * phi_p[i]) * fe_values.JxW(q); //explicit
                            
                        }
                        const unsigned int component_i = fe.system_to_component_index(i).first;                        
                        local_rhs(i) += (deltat*density1*(fe_values.shape_value(i,q) * rhs_values[q](component_i)) + density1*old_solution_values[q]*value_phi_u[i]) * fe_values.JxW(q); //implicit

//                         local_rhs(i) += deltat*density1*(fe_values.shape_value(i,q) * rhs_values[q](component_i) + density1*old_solution_values[q]*value_phi_u[i] - deltat*density1*((old_velocity_gradients[q]*(old_solution_values[q] - meshvelocity_values[q]))*value_phi_u[i])) * fe_values.JxW(q); //explicit
                    } // end of i loop                
                }  // end of quadrature points loop
                
//                 for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
//                 {
//                     if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 102))
//                     {             
//                         fe_face_values.reinit (cell, face_n);
//                         traction_boundary_values.vector_value_list(fe_face_values.get_quadrature_points(), neumann_boundary_values);
//                         for (unsigned int q=0; q<n_face_q_points; ++q)
//                             for (unsigned int i=0; i<dofs_per_cell; ++i)
//                             {                 
//                                 const unsigned int component_i = fe.system_to_component_index(i).first;
//                                 local_rhs(i) += (fe_face_values.shape_value(i, q) * neumann_boundary_values[q](component_i) * fe_face_values.JxW(q))*deltat/density1;
//                             }
//                     } // end of face if
//                 } // end of face for
                cell->get_dof_indices (local_dof_indices);         
                stokesconstraints.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, ns_system_matrix, lo_system_rhs);
            } // end of if cell->locally owned
            vof_cell++;
            cell_move++;
        } // end of cell loop
        ns_system_matrix.compress (VectorOperation::add);
        lo_system_rhs.compress (VectorOperation::add);
    }
    //=======================================
    template <int dim>
    void StokesProblem<dim>::assemble_vofsystem()
    {
        TimerOutput::Scope t(computing_timer, "assembly_vofsystem");
        pcout << "assemble_vofsystem" << std::endl;
        vof_system_matrix=0;
        lo_vof_system_rhs=0;
        
        QGauss<dim>   vof_quadrature_formula(degree+2);
//         QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEValues<dim> fe_vof_values (fevof, vof_quadrature_formula,
                                     update_values  |
                                     update_quadrature_points |
                                     update_JxW_values |
                                     update_gradients | update_hessians);
        
        FEValues<dim> fe_velocity_values (fe, vof_quadrature_formula,
                                          update_values    |
                                          update_quadrature_points  |
                                          update_JxW_values |
                                          update_gradients);
        
        FEValues<dim> fe_move_values (femove, vof_quadrature_formula, update_values);
        
                
//         FEFaceValues<dim> fe_face_values (fevof, face_quadrature_formula,
//                                           update_values    | update_normal_vectors |
//                                           update_quadrature_points  | update_JxW_values);
        
//         FEFaceValues<dim> fe_face_velocity_values (fe, face_quadrature_formula,
//                                           update_values | update_quadrature_points | update_JxW_values);
        
//         VofBoundaryValues<dim> vof_boundary_values;
        
        const unsigned int dofs_per_cell = fevof.dofs_per_cell;
        const unsigned int vof_n_q_points = vof_quadrature_formula.size();
//         const unsigned int n_face_q_points = face_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        
        FullMatrix<double>                   vof_local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>                       vof_local_rhs(dofs_per_cell);        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
//         const VofRightHandSide<dim>          vof_right_hand_side;
        std::vector<double>                  vof_rhs_values(vof_n_q_points);
        std::vector<double>                  old_vof_values(vof_n_q_points);
        std::vector<Tensor<1,dim>>           old_vof_gradients(vof_n_q_points);
//         std::vector<double>                  old_vof_face_values(n_face_q_points);
        std::vector<Tensor<1,dim>>           nsvelocity_values(vof_n_q_points);
//         std::vector<Tensor<1,dim>>           nsvelocity_values_face(n_face_q_points);
        std::vector<double>                  value_phi_vof(dofs_per_cell);
        std::vector<Tensor<1,dim>>           gradient_phi_vof(dofs_per_cell);
        std::vector<Tensor<1, dim> >         meshvelocity_values(vof_n_q_points);
        std::vector<Tensor<2,dim>>           hessian_phi_vof(dofs_per_cell);
        double taucell = 0.0;
        double velocity_norm = 0.0;
        double peclet_num = 0.0;
        typename DoFHandler<dim>::active_cell_iterator vof_cell = vof_dof_handler.begin_active(), vof_endc = vof_dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator cellns = dof_handler.begin_active();
        typename DoFHandler<dim>::active_cell_iterator cell_move = move_dof_handler.begin_active();        
//         pcout << "CFL is "<< vof_cell->diameter() << std::endl; //assuming max_vel is 1m/s
        
        for (; vof_cell!=vof_endc; ++vof_cell)
        {
            if (vof_cell->is_locally_owned())
            {
                fe_vof_values.reinit(vof_cell);
                fe_velocity_values.reinit(cellns);
                fe_move_values.reinit(cell_move);
//                 vof_right_hand_side.value_list(fe_vof_values.get_quadrature_points(), vof_rhs_values);
                fe_velocity_values[velocities].get_function_values(lr_solution, nsvelocity_values);
                fe_vof_values.get_function_values(lr_vof_solution, old_vof_values);
                fe_vof_values.get_function_gradients(lr_vof_solution, old_vof_gradients);
                fe_move_values[velocities].get_function_values(lr_move_solution, meshvelocity_values);
                vof_local_matrix = 0;
                vof_local_rhs = 0;

                for (unsigned int q_index=0; q_index<vof_n_q_points; ++q_index)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {                        
                        value_phi_vof[k]   = fe_vof_values.shape_value (k, q_index);
                        gradient_phi_vof[k]= fe_vof_values.shape_grad (k, q_index);
                        hessian_phi_vof[k] = fe_vof_values.shape_hessian (k, q_index);
                    }
                    
                velocity_norm = nsvelocity_values[q_index].norm();
//                 velocity_norm = (nsvelocity_values[q_index] - meshvelocity_values[q_index]).norm();
                if(temp == 0 || velocity_norm/temp > 1e6)
                {
                    taucell = vof_cell->diameter()/(2*velocity_norm);
                    taucell *= 2.0;
                }
                else
                {
                peclet_num = (velocity_norm*vof_cell->diameter()/2)/temp;
                taucell = vof_cell->diameter()/(2*velocity_norm)*(cosh(peclet_num)/sinh(peclet_num) - 1/peclet_num);
                taucell *= 2.0;
                }
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            vof_local_matrix(i,j) += (value_phi_vof[j]*(value_phi_vof[i] + taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]) -
                            deltat*(meshvelocity_values[q_index]*gradient_phi_vof[j])*(value_phi_vof[i]+taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]) -
                            deltat*value_phi_vof[j]*(nsvelocity_values[q_index]*gradient_phi_vof[i]) + deltat*taucell*(nsvelocity_values[q_index]*gradient_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i]) +
                            deltat*temp*(gradient_phi_vof[j]*gradient_phi_vof[i]) - deltat*temp*taucell*(trace(hessian_phi_vof[j])*(nsvelocity_values[q_index]*gradient_phi_vof[i]))
                            )*fe_vof_values.JxW(q_index);  //implicit

//                             vof_local_matrix(i,j) += (value_phi_vof[j]*(value_phi_vof[i] + taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]))*fe_vof_values.JxW(q_index);  //explicit
                        }
                        vof_local_rhs(i) += (old_vof_values[q_index]*(value_phi_vof[i] + taucell*(nsvelocity_values[q_index]*gradient_phi_vof[i])))*fe_vof_values.JxW(q_index); //implicit

//                             vof_local_rhs(i) += (old_vof_values[q_index]*(value_phi_vof[i] + taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]) +
//                             deltat*(meshvelocity_values[q_index]*old_vof_gradients[q_index])*(value_phi_vof[i]+taucell*nsvelocity_values[q_index]*gradient_phi_vof[i]) +
//                             deltat*old_vof_values[q_index]*(nsvelocity_values[q_index]*gradient_phi_vof[i]) - deltat*taucell*(nsvelocity_values[q_index]*old_vof_gradients[q_index])*(nsvelocity_values[q_index]*gradient_phi_vof[i]))*fe_vof_values.JxW(q_index);  //explicit. according to complete theory
                    }
                } //end of quadrature points loop
//                 } // end of face for
                
//                 for (unsigned int face_n=0;  face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
//                 {
//                  if (vof_cell->face(face_n)->at_boundary() && (vof_cell->face(face_n)->boundary_id() == 104))
//                     {             
//                         fe_face_values.reinit (vof_cell, face_n);
//                         fe_face_velocity_values.reinit (cellns, face_n);
//                         fe_face_velocity_values[velocities].get_function_values(lr_solution, nsvelocity_values_face);
// //                         fe_face_values.get_function_values(lr_vof_solution, old_vof_face_values);
//                         vof_boundary_values.value_list(fe_face_values.get_quadrature_points(), old_vof_face_values);
//                         
//                         for (unsigned int q=0; q<n_face_q_points; ++q)
//                             for (unsigned int i=0; i<dofs_per_cell; ++i)            
//                             {
//                                     vof_local_rhs(i) += -deltat*(old_vof_face_values[q]*value_phi_vof[i]*(nsvelocity_values_face[q]*fe_face_values.normal_vector(q)))*fe_face_values.JxW(q);
//                             }
//                     } // end of face if
//                     
//                     if (vof_cell->face(face_n)->at_boundary() && (vof_cell->face(face_n)->boundary_id() == 102))
//                     {             
//                         fe_face_values.reinit (vof_cell, face_n);
//                         fe_face_velocity_values.reinit (cellns, face_n);
//                         fe_face_velocity_values[velocities].get_function_values(lr_solution, nsvelocity_values_face);
//                         
//                         for (unsigned int q=0; q<n_face_q_points; ++q)
//                             for (unsigned int i=0; i<dofs_per_cell; ++i)            
//                             {
//                                 for (unsigned int j=0; j<dofs_per_cell; ++j)
//                                 {
// //                                     vof_local_rhs(i) += -deltat*(old_vof_face_values[q]*value_phi_vof[i]*(nsvelocity_values_face[q]*fe_face_values.normal_vector(q)))*fe_face_values.JxW(q);
//                                 vof_local_matrix(i,j) += deltat*(value_phi_vof[j]*value_phi_vof[i]*(nsvelocity_values_face[q]*fe_face_values.normal_vector(q)))*fe_face_values.JxW(q);
//                                 }
//                             }
//                     } // end of face if
//                 } // end of face for
                vof_cell->get_dof_indices(local_dof_indices);                
                vofconstraints.distribute_local_to_global(vof_local_matrix, vof_local_rhs, local_dof_indices, vof_system_matrix, lo_vof_system_rhs);     
            } //end of if vof_cell->is_locally_owned()
            cellns++;
            cell_move++;
        } //end of cell loop
        vof_system_matrix.compress (VectorOperation::add);
        lo_vof_system_rhs.compress (VectorOperation::add);
    }
      //============================================================
    template <int dim>
    void StokesProblem<dim>::assemble_stokessystem_nonlinear()
    {
        TimerOutput::Scope t(computing_timer, "assembly_system_nonlinear");
        pcout <<"in assemble_system_nonlinear"<<std::endl;
        ns_system_matrix=0;
        
        QGauss<dim>   quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |
                                 update_quadrature_points  |
                                 update_JxW_values |
                                 update_gradients);
        
        FEValues<dim> fe_move_values (femove, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
        
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim); 
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        std::vector<types::global_dof_index>  local_dof_indices (dofs_per_cell);        
        
        std::vector<Tensor<1, dim>>           value_phi_u (dofs_per_cell);    
        std::vector<Tensor<2, dim>>           gradient_phi_u (dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>>  symgrad_phi_u (dofs_per_cell);
        std::vector<double>                   div_phi_u(dofs_per_cell);
        std::vector<double>                   phi_p(dofs_per_cell);
        std::vector<Tensor<2, dim> >          solution_gradients(n_q_points);
        std::vector<Tensor<1, dim> >          old_solution_values(n_q_points);
        std::vector<Tensor<1, dim> >          meshvelocity_values(n_q_points);        
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator cell_move = move_dof_handler.begin_active();
        
        for (; cell!=endc; ++cell)
        {    
            if(cell->is_locally_owned())
            {
                fe_values.reinit (cell);
                fe_move_values.reinit(cell_move);
                fe_values[velocities].get_function_values(lr_old_iterate_solution, old_solution_values);
                fe_move_values[velocities].get_function_values(lr_move_solution, meshvelocity_values);
                local_matrix = 0;
                
                for (unsigned int q=0; q<n_q_points; ++q)
                {
                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                    {
                        value_phi_u[k]   = fe_values[velocities].value (k, q);
                        gradient_phi_u[k]= fe_values[velocities].gradient (k, q);
                        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient (k, q);
                        div_phi_u[k]     = fe_values[velocities].divergence (k, q);
                        phi_p[k]         = fe_values[pressure].value (k, q);
                    }
                    
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {                    
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            local_matrix(i, j) += ((value_phi_u[i]*value_phi_u[j] + 
                            deltat * value_phi_u[i] * (gradient_phi_u[j] * (old_solution_values[q] - meshvelocity_values[q])) +
//                             deltat * value_phi_u[i] * (gradient_phi_u[j] * (old_solution_values[q])) +
                            (2*deltat*viscosity/density)*scalar_product(symgrad_phi_u[i], symgrad_phi_u[j])) -
                            deltat * div_phi_u[i] * phi_p[j]/density - 
                            phi_p[i] * div_phi_u[j]) *
                            fe_values.JxW(q);
                        }
                    } // end of i loop
                }  // end of quadrature points loop                
                cell->get_dof_indices (local_dof_indices);
                stokesconstraints.distribute_local_to_global(local_matrix, local_dof_indices, ns_system_matrix);
            }
            cell_move++;
        } // end of cell loop
        ns_system_matrix.compress (VectorOperation::add);
        pcout <<"end of assemble_system_nonlinear "<<std::endl;
    } // end of assemble system
    //================================================================
    template <int dim>
    void StokesProblem<dim>::assemble_movemesh()
    {
        TimerOutput::Scope t(computing_timer, "assembly_movemesh");
        pcout << "assemble_movemesh" << std::endl;
        move_system_matrix=0;
        lo_move_system_rhs=0;
        QGauss<dim>   move_quadrature_formula(degree+2);
        
        FEValues<dim> move_fe_values (femove, move_quadrature_formula,
                                      update_values  |
                                      update_quadrature_points |
                                      update_JxW_values |
                                      update_gradients);        
        
        const unsigned int dofs_per_cell = femove.dofs_per_cell;
        const unsigned int move_n_q_points = move_quadrature_formula.size();
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        
        FullMatrix<double>                   move_local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>                       move_local_rhs(dofs_per_cell);        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        const MoveRightHandSide<dim>         move_right_hand_side;
        std::vector<Tensor<1,dim>>           move_rhs_values(move_n_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator cell = move_dof_handler.begin_active(), endc = move_dof_handler.end();
        
        for (; cell!=endc; ++cell)
        {
            if (cell->is_locally_owned())
            {
                move_fe_values.reinit(cell);
                move_local_matrix = 0;
                move_local_rhs = 0;
                move_right_hand_side.value_list(move_fe_values.get_quadrature_points(), move_rhs_values);
                
                for (unsigned int q_index=0; q_index<move_n_q_points; ++q_index)
                {
                    for (unsigned int i=0; i<dofs_per_cell; ++i)            
                    {
                        const Tensor<2,dim> symmgrad_i_u = move_fe_values[velocities].symmetric_gradient(i, q_index);
                        const double div_i_u = move_fe_values[velocities].divergence(i, q_index);                        
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        { 
                            const Tensor<2,dim> symmgrad_j_u = move_fe_values[velocities].symmetric_gradient(j, q_index);
                            const double div_j_u = move_fe_values[velocities].divergence(j, q_index);
                            move_local_matrix(i,j) += (scalar_product(symmgrad_i_u, symmgrad_j_u) + div_i_u*div_j_u) * move_fe_values.JxW(q_index);
                        }
                        move_local_rhs(i) += 0;
                    }
                }
                cell->get_dof_indices(local_dof_indices);                
                moveconstraints.distribute_local_to_global(move_local_matrix, move_local_rhs, local_dof_indices, move_system_matrix, lo_move_system_rhs);
            } // end of if cell->is_locally_owned()
        } //  end of cell loop   
        move_system_matrix.compress (VectorOperation::add);
        lo_move_system_rhs.compress (VectorOperation::add);
    }
    //================================================================
     template <int dim>
    void StokesProblem<dim>::compute_stresses()
    {
        TimerOutput::Scope t(computing_timer, "compute_stresses");
        pcout <<"compute_stresses "<<std::endl;

        QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values | update_gradients   | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        
        FEFaceValues<dim> fe_face_vof_values(fevof, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        std::vector<Point<dim>> quadrature_points(n_face_q_points);
        
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        
        std::vector<Tensor<2, dim>>           face_gradients(n_face_q_points);
        std::vector<double>                   face_vof_values(n_face_q_points);
        std::vector<double>                   face_pressure_values(n_face_q_points);        
        Tensor<1,dim>                         point_force;
        
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        typename DoFHandler<dim>::active_cell_iterator vof_cell = vof_dof_handler.begin_active();

        Tensor<2, dim> identity;
        identity[0][0] = 1;
        identity[1][1] = 1;
        
        local_force[0] = 0;
        local_force[1] = 0;
        local_moment = 0;
        
        for (; cell!=endc; ++cell)
        { 
            if (cell->is_locally_owned())
            {
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
                {
                    if (cell->face(face_n)->at_boundary() && (cell->face(face_n)->boundary_id() == 105))
                    {
                        fe_face_values.reinit (cell, face_n);
                        fe_face_vof_values.reinit(vof_cell, face_n);
                        fe_face_values[velocities].get_function_gradients(lr_solution, face_gradients);
                        fe_face_values[pressure].get_function_values(lr_solution, face_pressure_values);
                        fe_face_vof_values.get_function_values(lr_vof_solution, face_vof_values);
                        for (unsigned int q=0; q<n_face_q_points; ++q)
                        {
                            quadrature_points = fe_face_values.get_quadrature_points();
                            
                            local_force += ((-2*((face_vof_values[q]*viscosity)+(1-face_vof_values[q])*viscosity_air)*((face_gradients[q] + transpose(face_gradients[q]))) + (face_pressure_values[q]*identity))*fe_face_values.normal_vector(q))*fe_face_values.JxW(q);

                            point_force = (-2*((face_vof_values[q]*viscosity)+(1-face_vof_values[q])*viscosity_air)*((face_gradients[q] + transpose(face_gradients[q]))) + (face_pressure_values[q]*identity))*fe_face_values.normal_vector(q);
                            
                            local_moment += ((quadrature_points[q][0] - centre_of_mass[0])*point_force[1] - (quadrature_points[q][1] - centre_of_mass[1])*point_force[0])*fe_face_values.JxW(q);
                        }
                    } // end of face if
                } // end of face for
            } // end of if cell->locally owned
            vof_cell++;
        } // end of cell loop
        
        MPI_Allreduce(&local_force[0], &total_force[0], 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        MPI_Allreduce(&local_force[1], &total_force[1], 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        MPI_Allreduce(&local_moment, &total_moment, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }
        //=================================================================
    template <int dim>
    void StokesProblem<dim>::compute_rigidbody_vel()
    {
        prev_vel_rbody = vel_rbody;
        total_force[1] = total_force[1] - mass_rigidbody*10; //-m*g in y dir(weight)
//         total_force[0] = 0.0;
//         total_force[1] = 0.0;
//         total_moment = 0.0;
        vel_rbody += deltat*total_force/(mass_rigidbody*(density_rbody/5));
        rotation_vel_rbody += deltat*total_moment/(moment_of_inertia*(density_rbody*1.15));
//         vel_rbody[1] = 0;
//         rotation_vel_rbody = 0;
        pcout << "total_force[0] " << total_force[0] << std::endl;
        pcout << "total_force[1] " << total_force[1] << std::endl;
        pcout << "total_moment " << total_moment << std::endl;
        
        pcout << "vel_rbody[1] " << vel_rbody[1] << std::endl;
//         pcout << "rotation_vel_rbody " << rotation_vel_rbody << std::endl;
        
        total_force[0] = 0.0;
        total_force[1] = 0.0;
        total_moment = 0.0;
    }
    //===============================================
    template <int dim>
    void StokesProblem<dim>::solve_stokes()
    {
        pcout <<"solve_stokes"<<std::endl;
        TimerOutput::Scope t(computing_timer, "solve_stokes");
        LA::MPI::Vector  distributed_solution_stokes (owned_partitioning_stokes, mpi_communicator);    
        double nonlinear_error = 1e5;
        int nonlinear_iterations = 0;
        SolverControl solver_control_stokes (dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_stokes(solver_control_stokes, mpi_communicator);        
        
        while(nonlinear_error > 1e-4)
        {
        solver_stokes.solve (ns_system_matrix, distributed_solution_stokes, lo_system_rhs);
        stokesconstraints.distribute(distributed_solution_stokes);
        lr_solution = distributed_solution_stokes;
//         distributed_solution_stokes -= lr_old_iterate_solution;
//         lr_nonlinear_residue = distributed_solution_stokes;
        
        nonlinear_error = compute_nonlinear_residue();
        pcout <<"residual_norm is " << nonlinear_error << std::endl;  
        lr_old_iterate_solution = lr_solution;
        pcout <<"end of nonlinear iteration "<< nonlinear_iterations << std::endl;
        nonlinear_iterations++;
        if(nonlinear_error < 1e-4 || nonlinear_iterations > 3)
            break;
        assemble_stokessystem_nonlinear();            
        } // end of while
    }
    //===============================================
    template <int dim>
    void StokesProblem<dim>::solve_vof()
    {
        pcout <<"solve_vof"<<std::endl;
        TimerOutput::Scope t(computing_timer, "solve_vof");
        LA::MPI::Vector  distributed_solution_vof_adjusted (owned_partitioning_vof, mpi_communicator);        
        SolverControl solver_control_vof (vof_dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_vof(solver_control_vof, mpi_communicator);       
        solver_vof.solve (vof_system_matrix, distributed_solution_vof_adjusted, lo_vof_system_rhs);
        
        for(unsigned int i = distributed_solution_vof_adjusted.local_range().first; i < distributed_solution_vof_adjusted.local_range().second; ++i)
        {
            if(distributed_solution_vof_adjusted(i) > 1.05)
                distributed_solution_vof_adjusted(i) = 1.05;
            else if(distributed_solution_vof_adjusted(i) < -1.05)
                distributed_solution_vof_adjusted(i) = -1.05;
        }
        distributed_solution_vof_adjusted.compress(VectorOperation::insert);
        
        vofconstraints.distribute(distributed_solution_vof_adjusted);
        lr_vof_solution = distributed_solution_vof_adjusted;
    }
    //===============================================
    template <int dim>
    void StokesProblem<dim>::solve_movemesh()
    {
        pcout <<"solve_movemesh"<<std::endl;
        TimerOutput::Scope t(computing_timer, "solve_movemesh");
        LA::MPI::Vector  distributed_solution_movemesh (owned_partitioning_movemesh, mpi_communicator);
        SolverControl solver_control_movemesh (move_dof_handler.n_dofs(), 1e-12);
        dealii::PETScWrappers::SparseDirectMUMPS solver_movemesh(solver_control_movemesh, mpi_communicator);
        
        solver_movemesh.solve (move_system_matrix, distributed_solution_movemesh, lo_move_system_rhs);
        moveconstraints.distribute(distributed_solution_movemesh);
        lr_move_solution = distributed_solution_movemesh;
        lo_total_meshmove += lr_move_solution;
    }
    //===============================================
    template <int dim>
    void StokesProblem<dim>::movemesh()
    {
        pcout << "moving mesh..." << std::endl;
        lo_total_meshmove *= 0.5*deltat;
        lr_total_meshmove = lo_total_meshmove;
        
        std::vector<bool> vertex_touched(triangulation.n_vertices(), false);        
        Point<dim> vertex_displacement;
        const std::vector<bool> vertex_locally_moved = GridTools::get_locally_owned_vertices(triangulation);
        
        for (typename DoFHandler<dim>::active_cell_iterator  cell = move_dof_handler.begin_active(); cell != move_dof_handler.end(); ++cell)
        {
            if(cell->is_locally_owned())
                for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                    if (vertex_touched[cell->vertex_index(v)] == false)
                    {
                        vertex_touched[cell->vertex_index(v)] = true;
                                            for (unsigned int d=0; d<dim; ++d)
                                                vertex_displacement[d] = lr_total_meshmove(cell->vertex_dof_index(v,d));

                        cell->vertex(v) += vertex_displacement;
                    }
        }
        triangulation.communicate_locally_moved_vertices(vertex_locally_moved);
        lo_total_meshmove = lr_move_solution;
        
        centre_of_mass[1] += (prev_vel_rbody[1] + vel_rbody[1])*deltat/2; //update y-coord of centre_of_mass
    }
    //=================================================
    template <int dim>
    double StokesProblem<dim>::compute_linf_vof()
    {      
        Vector<double> cellwise_errors(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (vof_dof_handler, lr_vof_solution, ZeroFunction<dim>(1), cellwise_errors, quadrature, VectorTools::Linfty_norm);
        const double vof_linf = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::Linfty_norm);
        return vof_linf;
    }
    //====================================================
    template <int dim>
    double StokesProblem<dim>::compute_maxvel()
    {        
        const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim), dim+1);        
        Vector<double> cellwise_errors(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (dof_handler, lr_solution, ZeroFunction<dim>(dim+1), cellwise_errors, quadrature, VectorTools::Linfty_norm, &velocity_mask);
        const double max_vel = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::Linfty_norm);
        return max_vel;
    }
    //====================================================
    template <int dim>
    void StokesProblem<dim>::compute_hmin()
    {
        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
        
        for (; cell!=endc; ++cell)
        { 
            if (cell->is_locally_owned())
            {
                if(cell->diameter() < local_hmin)
                    local_hmin = cell->diameter();         
            }
        }
        MPI_Allreduce(&local_hmin, &global_hmin, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);
    }
        //==========================================
    template <int dim>
    double StokesProblem<dim>::compute_nonlinear_residue()
    {        
        const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0,dim), dim+1);
        ns_system_matrix.vmult(lo_lhs_productvec, lr_old_iterate_solution);
        lo_lhs_productvec -= lo_system_rhs;
        lr_nonlinear_residue = lo_lhs_productvec;
        Vector<double> cellwise_errors(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (dof_handler, lr_nonlinear_residue, ZeroFunction<dim>(dim+1), cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
        const double residue_norm = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);
        return residue_norm;
    }
    //=================================================
    template <int dim>
    void StokesProblem<dim>::compute_masses()
    {      
        Vector<double> cellwise_fluid_fractions(triangulation.n_active_cells());
        Vector<double> cellwise_gas_fractions(triangulation.n_active_cells());
        QGauss<dim> quadrature(4);
        VectorTools::integrate_difference (vof_dof_handler, lr_vof_solution, ZeroFunction<dim>(1), cellwise_fluid_fractions, quadrature, VectorTools::mean);
        VectorTools::integrate_difference (vof_dof_handler, lr_vof_solution, ConstantFunction<dim>(1, 1), cellwise_gas_fractions, quadrature, VectorTools::mean);
        const double fluid_mass = VectorTools::compute_global_error(triangulation, cellwise_fluid_fractions, VectorTools::mean);
        const double gas_mass = VectorTools::compute_global_error(triangulation, cellwise_gas_fractions, VectorTools::mean);
        pcout << "fluid_mass is " << -fluid_mass << std::endl;
        pcout << "gas_mass is " << gas_mass << std::endl;
    }
    //===================================================================
    template <int dim>
    void StokesProblem<dim>::output_results(int timestepnumber)
    {
        TimerOutput::Scope t(computing_timer, "output");
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.emplace_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (lr_solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);
        
        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i=0; i<subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector (subdomain, "subdomain");
        data_out.add_data_vector(vof_dof_handler, lr_vof_solution, "vof");
        data_out.build_patches ();
        
        std::string filenamebase = "zfs2d_250rho_node1";
        
        const std::string filename = (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." +Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
        
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::vector<std::string> filenames;
            for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
                filenames.push_back (filenamebase + Utilities::int_to_string (timestepnumber, 3) + "." + Utilities::int_to_string (i, 4) + ".vtu");
            
            std::ofstream master_output ((filenamebase + Utilities::int_to_string (timestepnumber, 3) + ".pvtu").c_str());
            data_out.write_pvtu_record (master_output, filenames);
        }
    }
    //==================================================================  
    template <int dim>
    void StokesProblem<dim>::timeloop()
    {      
        double timet = deltat;
        int timestepnumber=0;
        BoundaryValues105<dim> boundaryvalues105;
        moveBoundaryValues105<dim> moveboundaryvalues105;
        RightHandSide<dim> righthandside;
        double max_vel = 0.0;
        
        while(timet<totaltime)
        {
            pcout << "==============================" << std::endl;
            righthandside.set_time(timet);
            output_results(timestepnumber);
//             exit(0);
            boundaryvalues105.set_rbody_vels(vel_rbody, rotation_vel_rbody);
            boundaryvalues105.set_centre_of_mass(centre_of_mass);
            resetup_stokessystem(boundaryvalues105);
            assemble_stokessystem(righthandside);
            solve_stokes();
            max_vel = compute_maxvel();
            pcout << "maxxxxxxx_vellllllllll = " << max_vel << std::endl;
            assemble_vofsystem();
            solve_vof();
            if(timestepnumber%20 == 0)
                compute_masses();
            pcout<<"linf_vof = " << compute_linf_vof() << std::endl;
            compute_stresses();
            compute_rigidbody_vel();
            moveboundaryvalues105.set_rbody_vels(vel_rbody, rotation_vel_rbody);
            moveboundaryvalues105.set_centre_of_mass(centre_of_mass);
            resetup_movemeshsystem(moveboundaryvalues105);
            assemble_movemesh();
            solve_movemesh();
            movemesh();
//             compute_hmin();
//             pcout << "global_hmin " << global_hmin << std::endl;
//             deltat = (max_vel<1e-6)?1e-6:global_hmin/max_vel;
//             pcout << "deltat " << deltat << std::endl;
//             if(timet >= 0.1)
//                 deltat = 0.0025;
            pcout <<"timet "<<timet <<std::endl;
            timet+=deltat;
            pcout << "timestepnumber " << timestepnumber++ << std::endl;
        }
        output_results(timestepnumber);
    }
}  // end of namespace
//====================================================
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Navierstokes;        
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);        
        StokesProblem<2> flow_problem(1);   
        flow_problem.setup_stokessystem();
        flow_problem.setup_vofsystem();
        flow_problem.setup_movemeshsystem();
        flow_problem.timeloop();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }    
    return 0;
}
