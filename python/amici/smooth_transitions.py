"""
Smooth transitions
------------
This module provides helper functions for reading/writing smooth transitions
with AMICI annotations from/to SBML files
and for adding such functions to the AMICI C++ code.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sbml_import import SbmlImporter
    from typing import (
        Union,
        Optional,
        Dict,
        Any,
        List,
    )

import xml.etree.ElementTree as ET
import numpy as np
import sympy as sp
import libsbml

from sympy.core.parameters import evaluate

from .sbml_utils import (
    sbml_time_symbol,
    amici_time_symbol,
    pretty_xml,
    mathml2sympy,
    sbmlMathML,
    annotation_namespace,
    hasParameter,
    addParameter,
    addAssignmentRule,
)


class SmoothTransition:
    def __init__(
            self,
            sbmlId: Union[str, sp.Symbol],
            x: sp.Basic,
            x0: Union[float, sp.Basic],
            dx: Union[float, sp.Basic],
            y0: Union[float, sp.Basic],
            y1: Union[float, sp.Basic]
        ):

        self._sbmlId = sbmlId
        self._x = sp.sympify(x)
        self._x0 = sp.sympify(x0)
        self._dx = sp.sympify(dx)
        self._y0 = sp.sympify(y0)
        self._y1 = sp.sympify(y1)

    @property
    def sbmlId(self):
        return self._sbmlId

    @property
    def x(self):
        return self._x

    @property
    def x0(self):
        return self._x0

    @property
    def dx(self):
        return self._dx

    @property
    def y0(self):
        return self._y0

    @property
    def y1(self):
        return self._y1

    def replace_in_all_expressions(self, old, new):
        if self._sbmlId == old:
            self._sbmlId = new
        self._x = self._x.subs(old, new)
        self._x0 = self._x0.subs(old, new)
        self._dx = self._dx.subs(old, new)
        self._y0 = self._y0.subs(old, new)
        self._y1 = self._y1.subs(old, new)

    @property
    def formula(self) -> sp.Piecewise:
        """
        Compute a symbolic formula for the smooth transition.
        """
        return self._formula(sbml=False)

    @property
    def sbmlFormula(self) -> sp.Piecewise:
        """
        Compute a symbolic formula for the smooth transition,
        using SBML symbol naming
        (the AMICI time symbol will be replaced with its SBML counterpart).
        """
        return self._formula(sbml=True)

    @property
    def mathmlFormula(self) -> sp.Piecewise:
        """
        Compute a symbolic formula for the smooth transition
        for use inside a SBML assignment rule: SBML symbol naming will be used
        and operations not supported by SBML MathML will be avoided.
        """
        return self._formula(sbml=True)

    def _formula(self, *, x=None, sbml=False) -> sp.Piecewise:
        if x is None:
            x = self.x

        s = (x - self.x0) / self.dx

        formula = sp.Piecewise(
            (self.y0, x < self.x0),
            (self.y0 + (self.y1 - self.y0) * s**2 * (3 - 2*s), x < self.x0 + self.dx),
            (self.y1, True)
        )

        if sbml:
            return formula.subs(amici_time_symbol, sbml_time_symbol)
        else:
            return formula

    @property
    def amiciAnnotation(self) -> str:
        """
        An SBML annotation describing the smooth transition.
        """
        annotation = f'<amici:smooth_transition xmlns:amici="{annotation_namespace}"'

        for (attr, value) in self._annotation_attributes().items():
            if isinstance(value, bool):
                value = str(value).lower()
            value = f'"{value}"'
            annotation += f' amici:{attr}={value}'
        annotation += '>'

        for (child, grandchildren) in self._annotation_children().items():
            if isinstance(grandchildren, str):
                grandchildren = [grandchildren]
            annotation += f'<amici:{child}>'
            for gc in grandchildren:
                annotation += gc
            annotation += f'</amici:{child}>'

        annotation += '</amici:smooth_transition>'

        # Check XML and prettify
        return pretty_xml(annotation)

    def _annotation_attributes(self) -> Dict[str, Any]:
        return {}

    def _annotation_children(self) -> Dict[str, Union[str, List[str]]]:
        children = {}

        with evaluate(False):
            x = self.x.subs(amici_time_symbol, sbml_time_symbol)
        children['evaluation_point'] = sbmlMathML(x)

        with evaluate(False):
            x0 = self.x0.subs(amici_time_symbol, sbml_time_symbol)
        children['transition_start'] = sbmlMathML(x0)

        with evaluate(False):
            dx = self.dx.subs(amici_time_symbol, sbml_time_symbol)
        children['transition_length'] = sbmlMathML(dx)

        with evaluate(False):
            y0 = self.y0.subs(amici_time_symbol, sbml_time_symbol)
        children['transition_start_value'] = sbmlMathML(y0)

        with evaluate(False):
            y1 = self.y1.subs(amici_time_symbol, sbml_time_symbol)
        children['transition_end_value'] = sbmlMathML(y1)

        return children

    def addToSbmlModel(self, model: libsbml.Model, *, auto_add: bool = True, units: Optional[str]):
        """
        Add the smooth transition to an SBML model using an assignment rule
        with AMICI-specific annotations.
        """
        if auto_add and not hasParameter(model, self.sbmlId):
            addParameter(model, self.sbmlId, constant=False, units=units)
        # Create assignment rule
        rule = addAssignmentRule(model, self.sbmlId, self.mathmlFormula)
        # Add annotation
        rule.setAnnotation(self.amiciAnnotation)

    @staticmethod
    def isSmoothTransition(rule: libsbml.AssignmentRule):
        """
        Determine if a SBML AssignmentRule
        is an AMICI-annotated smooth transition formula.
        """
        return SmoothTransition.getAnnotation(rule) is not None

    @staticmethod
    def getAnnotation(rule: libsbml.AssignmentRule):
        if not isinstance(rule, libsbml.AssignmentRule):
            raise TypeError('rule must be an AssignmentRule')
        if rule.isSetAnnotation():
            annotation = ET.fromstring(rule.getAnnotationString())
            for child in annotation.getchildren():
                if child.tag == f'{{{annotation_namespace}}}smooth_transition':
                    return child
        return None

    @staticmethod
    def fromAnnotation(sbmlId: sp.Symbol, annotation: ET.Element, *, locals):
        """
        Create a `SmoothTransition` object from a SBML annotation.
        """

        if annotation.tag != f'{{{annotation_namespace}}}smooth_transition':
            raise ValueError(
                'The given annotation is not an AMICI SBML annotation.'
            )

        attributes = {}
        for key, value in annotation.items():
            if not key.startswith(f'{{{annotation_namespace}}}'):
                raise ValueError(
                    f'unexpected attribute {key} inside AMICI annotation'
                )
            key = key[len(annotation_namespace)+2:]
            if value == 'true':
                value = True
            elif value == 'false':
                value = False
            attributes[key] = value

        children = {}
        for child in annotation.getchildren():
            if not child.tag.startswith(f'{{{annotation_namespace}}}'):
                raise ValueError(
                    f'unexpected node {child.tag} inside AMICI annotation'
                )
            key = child.tag[len(annotation_namespace)+2:]
            value = [
                mathml2sympy(
                    ET.tostring(gc).decode(),
                    evaluate=False,
                    locals=locals,
                    expression_type='Rule'
                )
                for gc in child.getchildren()
            ]
            children[key] = value

        kwargs = SmoothTransition._fromAnnotation(attributes, children)

        if len(attributes) != 0:
            raise ValueError(
                'Unprocessed attributes in AMICI annotation:\n' +
                str(attributes)
            )

        if len(children) != 0:
            raise ValueError(
                'Unprocessed children in AMICI annotation:\n' +
                str(children)
            )

        return SmoothTransition(sbmlId, **kwargs)

    @classmethod
    def _fromAnnotation(cls, attributes, children):
        kwargs = {}

        if 'evaluation_point' not in children.keys():
            raise ValueError(
                "required annotation 'evaluation_point' missing"
            )
        x = children.pop('evaluation_point')
        if len(x) != 1:
            raise ValueError(
                "Ill-formatted annotation 'evaluation_point' " +
                "(more than one children is present)"
            )
        kwargs['x'] = x[0]

        if 'transition_start' not in children.keys():
            raise ValueError(
                "required annotation 'transition_start' missing"
            )
        x0 = children.pop('transition_start')
        if len(x0) != 1:
            raise ValueError(
                "Ill-formatted annotation 'transition_start' " +
                "(more than one children is present)"
            )
        kwargs['x0'] = x0[0]

        if 'transition_length' not in children.keys():
            raise ValueError(
                "required annotation 'transition_length' missing"
            )
        dx = children.pop('transition_length')
        if len(dx) != 1:
            raise ValueError(
                "Ill-formatted annotation 'transition_length' " +
                "(more than one children is present)"
            )
        kwargs['dx'] = dx[0]

        if 'transition_start_value' not in children.keys():
            raise ValueError(
                "required annotation 'transition_start_value' missing"
            )
        y0 = children.pop('transition_start_value')
        if len(y0) != 1:
            raise ValueError(
                "Ill-formatted annotation 'transition_start_value' " +
                "(more than one children is present)"
            )
        kwargs['y0'] = y0[0]

        if 'transition_end_value' not in children.keys():
            raise ValueError(
                "required annotation 'transition_end_value' missing"
            )
        y1 = children.pop('transition_end_value')
        if len(y1) != 1:
            raise ValueError(
                "Ill-formatted annotation 'transition_end_value' " +
                "(more than one children is present)"
            )
        kwargs['y1'] = y1[0]

        return kwargs

    @property
    def odeModelSymbol(self):
        return self.y0 + (self.y1 - self.y0) * smooth_transition_cubic((self.x - self.x0) / self.dx)


class smooth_transition_cubic(sp.Function):
    nargs = (1, )

    @classmethod
    def eval(cls, *args):
        return None  # means leave unevaluated

    def fdiff(self, argindex=1):
        assert argindex == 1
        assert len(self.args) == 1
        return Dsmooth_transition_cubic(*self.args)

    def _eval_is_real(self):
        return True


class Dsmooth_transition_cubic(sp.Function):
    nargs = (1, )

    @classmethod
    def eval(cls, *args):
        return None  # means leave unevaluated

    def fdiff(self, argindex=1):
        raise NotImplementedError(
            'second derivative for smooth transition not implemented yet.'
        )

    def _eval_is_real(self):
        return True
