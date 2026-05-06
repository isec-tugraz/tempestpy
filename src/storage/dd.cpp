#include "dd.h"
#include "storm/storage/dd/DdManager.h"
#include "storm/storage/dd/DdMetaVariable.h"
#include "storm/storage/dd/Dd.h"
#include "storm/storage/dd/Bdd.h"
#include "storm/storage/dd/Add.h"
#include "src/helpers.h"

#include <map>


template<storm::dd::DdType DdType, typename ValueType>
std::map<storm::expressions::Variable, int_fast64_t> parse_add_valuation(storm::dd::Add<DdType, ValueType> const& add, py::dict const& valuation) {
    std::map<std::string, storm::expressions::Variable> variableByName;
    for (auto const& variable : add.getContainedMetaVariables()) {
        variableByName.emplace(variable.getName(), variable);
    }

    std::map<storm::expressions::Variable, int_fast64_t> result;
    for (auto item : valuation) {
        storm::expressions::Variable variable;
        if (py::isinstance<py::str>(item.first)) {
            std::string variableName = py::cast<std::string>(item.first);
            auto variableIt = variableByName.find(variableName);
            if (variableIt == variableByName.end()) {
                throw py::key_error("ADD does not contain meta variable '" + variableName + "'.");
            }
            variable = variableIt->second;
        } else {
            variable = py::cast<storm::expressions::Variable>(item.first);
            if (!add.containsMetaVariable(variable)) {
                throw py::key_error("ADD does not contain meta variable '" + variable.getName() + "'.");
            }
        }
        result[variable] = py::cast<int_fast64_t>(item.second);
    }
    return result;
}

template<storm::dd::DdType DdType, typename ValueType>
void define_add(py::module& m, py::class_<storm::dd::Dd<DdType>>& dd, std::string const& libstring, std::string const& valueTypeString) {
    py::class_<storm::dd::Add<DdType, ValueType>> add(m, (std::string("Add_") + libstring + "_" + valueTypeString).c_str(), "Add", dd);
    add.def("__iter__", [](storm::dd::Add<DdType, ValueType> const& s) { return py::make_iterator(s.begin(), s.end()); },
            py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);
    add.def("get_value", [](storm::dd::Add<DdType, ValueType> const& add, py::dict const& valuation) {
        return add.getValue(parse_add_valuation(add, valuation));
    }, py::arg("valuation") = py::dict(), "Evaluate the ADD for the given meta-variable valuation.");
    add.def("query", [](storm::dd::Add<DdType, ValueType> const& add, py::dict const& valuation) {
        return add.getValue(parse_add_valuation(add, valuation));
    }, py::arg("valuation"), "Evaluate the ADD for a valuation, e.g. add.query({'x': 4, 'y': 7}).");

    py::class_<storm::dd::AddIterator<DdType, ValueType>> addIterator(m, (std::string("AddIterator_") + libstring + "_" + valueTypeString).c_str(), "AddIterator");
    addIterator.def("get", [](storm::dd::AddIterator<DdType, ValueType> const& it) { return *it; });
}

template<storm::dd::DdType DdType>
void define_dd(py::module& m, std::string const& libstring) {
    py::class_<storm::dd::DdMetaVariable<DdType>> ddMetaVariable(m, (std::string("DdMetaVariable_") + libstring).c_str());
    ddMetaVariable.def("compute_indices", &storm::dd::DdMetaVariable<DdType>::getIndices, py::arg("sorted")=true);
    ddMetaVariable.def("can_represent", &storm::dd::DdMetaVariable<DdType>::canRepresent, py::arg("value"));
    ddMetaVariable.def_property_readonly("name", &storm::dd::DdMetaVariable<DdType>::getName);
    ddMetaVariable.def_property_readonly("lowest_value", &storm::dd::DdMetaVariable<DdType>::getLow);
    ddMetaVariable.def_property_readonly("has_high", &storm::dd::DdMetaVariable<DdType>::hasHigh);
    ddMetaVariable.def_property_readonly("highest_value", &storm::dd::DdMetaVariable<DdType>::getHigh);
    ddMetaVariable.def_property_readonly("nr_dd_variables", &storm::dd::DdMetaVariable<DdType>::getNumberOfDdVariables);
    ddMetaVariable.def_property_readonly("lowest_index", &storm::dd::DdMetaVariable<DdType>::getLowestIndex);
    ddMetaVariable.def_property_readonly("highest_level", &storm::dd::DdMetaVariable<DdType>::getHighestLevel);
    ddMetaVariable.def_property_readonly("indices_and_levels", &storm::dd::DdMetaVariable<DdType>::getIndicesAndLevels);
    ddMetaVariable.def_property_readonly("type", &storm::dd::DdMetaVariable<DdType>::getType);
    ddMetaVariable.def("__str__", &storm::dd::DdMetaVariable<DdType>::getName);

    py::class_<storm::dd::DdManager<DdType>, std::shared_ptr<storm::dd::DdManager<DdType>>> ddManager(m, (std::string("DdManager_") + libstring).c_str());
    ddManager.def("get_meta_variable", [](storm::dd::DdManager<DdType> const& manager, storm::expressions::Variable const& var) {return manager.getMetaVariable(var);}, py::arg("expression_variable"));

    py::class_<storm::dd::Dd<DdType>> dd(m, (std::string("Dd_") + libstring).c_str(), "Dd");
    dd.def_property_readonly("node_count", &storm::dd::Dd<DdType>::getNodeCount, "get node count");
    dd.def_property_readonly("dd_manager", &storm::dd::Dd<DdType>::getDdManager, "get the manager");
    dd.def_property_readonly("meta_variables", [](storm::dd::Dd<DdType> const& dd) {return dd.getContainedMetaVariables();}, "the contained meta variables");
    dd.def_property_readonly("meta_variable_names", [](storm::dd::Dd<DdType> const& dd) {
        std::vector<std::string> result;
        for (auto const& variable : dd.getContainedMetaVariables()) {
            result.push_back(variable.getName());
        }
        return result;
    }, "names of the contained meta variables");
    dd.def("get_meta_variable", [](storm::dd::Dd<DdType> const& dd, std::string const& name) {
        for (auto const& variable : dd.getContainedMetaVariables()) {
            if (variable.getName() == name) {
                return dd.getDdManager().getMetaVariable(variable);
            }
        }
        throw py::key_error("DD does not contain meta variable '" + name + "'.");
    }, py::arg("name"), "Get metadata for a contained meta variable by name.");



    py::class_<storm::dd::Bdd<DdType>> bdd(m, (std::string("Bdd_") + libstring).c_str(), "Bdd", dd);
    bdd.def("to_expression", &storm::dd::Bdd<DdType>::toExpression, py::arg("expression_manager"));

    define_add<DdType, double>(m, dd, libstring, "Double");
    define_add<DdType, storm::RationalNumber>(m, dd, libstring, "Exact");

}


void define_dd_nt(py::module& m) {
    py::enum_<storm::dd::MetaVariableType>(m, "DdMetaVariableType")
            .value("Int", storm::dd::MetaVariableType::Int)
            .value("Bool", storm::dd::MetaVariableType::Bool)
            .value("Bitvector", storm::dd::MetaVariableType::BitVector);

}

template void define_dd<storm::dd::DdType::Sylvan>(py::module& m, std::string const& libstring);
