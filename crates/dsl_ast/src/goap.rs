//! Goal-oriented action planning — real backward-chained precondition
//! search, resolved ENTIRELY at compile time, for every agent, on GPU.
//!
//! THE CORE IDEA: a `goap` block's fact/action/goal graph is static — fully
//! known from source text, not from any agent's live state. So the
//! expensive part (search) runs exactly ONCE, here, at compile time,
//! exploring the graph backward from the goal the same way a classic
//! regressive GOAP planner would (cheapest-producer-per-fact, cycle-
//! checked). What survives into the compiled kernel is NOT a search — it's
//! the search's OUTPUT, specialized into a plain nested `if`/`else`
//! EXPRESSION tree: "is this fact already true for me right now; if not,
//! are its prerequisites already true for me right now; if so, commit to
//! the action that produces it; if not, recurse into the first unmet one."
//! Every leaf reads only per-agent CURRENT field values (through each
//! fact's own boolean expression), so two agents in different states take
//! different branches through the SAME compiled tree — genuine plan-
//! directed, per-agent behavior — using nothing but ordinary `if`-as-
//! expression and `let`, both already fully supported by the existing
//! per-agent physics lowering. No new runtime GPU primitive (a mutable
//! loop-carried local, a dynamic array, an open-list) is needed at all.
//!
//! This module runs as a pure AST-to-AST desugaring pass, called from
//! `parser::parse_program` right after all top-level decls are collected —
//! by the time resolution (`resolve.rs`) ever sees `program.decls`, every
//! `Decl::Goap` has already been replaced with an ordinary `Decl::Physics`.
//! The rest of the compiler pipeline (IR, CG lowering, WGSL emission) has
//! no `Goap`-specific code at all — it only ever sees plain physics rules.

use crate::ast::{
    Annotation, AnnotationArg, AnnotationValue, CallArg, Decl, Expr, ExprKind, EventPattern,
    GoapActionDecl, GoapDecl, PhysicsDecl, PhysicsHandler, PhysicsPattern, Span, Stmt,
};
use std::collections::{HashMap, HashSet};

/// A desugaring failure — reported through the same `ParseError` surface as
/// every other parse-time error (see `parser::parse_program`'s call site).
#[derive(Debug)]
pub struct GoapDesugarError {
    pub span: Span,
    pub message: String,
}

/// Replace every `Decl::Goap` in `decls` with the `Decl::Physics` its graph
/// desugars to, in place. Errors point at the offending `goap`/`action`/
/// `fact`/`goal` span.
pub fn desugar_goap_decls(decls: &mut [Decl]) -> Result<(), GoapDesugarError> {
    for decl in decls.iter_mut() {
        if let Decl::Goap(g) = decl {
            let physics = desugar_one(g)?;
            *decl = Decl::Physics(physics);
        }
    }
    Ok(())
}

/// Per-fact planning info computed once, at compile time: the cheapest
/// action that produces this fact (`None` for a terminal fact — nothing
/// plans it, it's just checked directly) and its total cost from a cold
/// start (used only for cycle detection's "currently visiting" guard —
/// the emitted decision tree doesn't need the numeric cost at runtime,
/// since it always re-checks the LIVE fact state, not a cached plan cost).
struct FactPlan<'a> {
    producer: Option<&'a GoapActionDecl>,
}

fn desugar_one(g: &GoapDecl) -> Result<PhysicsDecl, GoapDesugarError> {
    let span = g.span;

    // ---- validate: unique fact names ----
    let mut fact_exprs: HashMap<&str, &Expr> = HashMap::new();
    for f in &g.facts {
        if fact_exprs.insert(f.name.as_str(), &f.expr).is_some() {
            return Err(GoapDesugarError {
                span: f.span,
                message: format!("goap `{}`: duplicate fact `{}`", g.name, f.name),
            });
        }
    }
    let known_fact = |name: &str| fact_exprs.contains_key(name);

    // ---- validate: every requires/produces name is a declared fact ----
    for a in &g.actions {
        for r in a.requires.iter().chain(a.produces.iter()) {
            if !known_fact(r) {
                return Err(GoapDesugarError {
                    span: a.span,
                    message: format!(
                        "goap `{}`: action `{}` references undeclared fact `{r}`",
                        g.name, a.name
                    ),
                });
            }
        }
    }
    for r in &g.goal.requires {
        if !known_fact(r) {
            return Err(GoapDesugarError {
                span: g.goal.span,
                message: format!("goap `{}`: goal references undeclared fact `{r}`", g.name),
            });
        }
    }

    // ---- validate: unique action ids ----
    let mut seen_ids: HashSet<i64> = HashSet::new();
    for a in &g.actions {
        if !seen_ids.insert(a.id) {
            return Err(GoapDesugarError {
                span: a.span,
                message: format!(
                    "goap `{}`: action `{}` reuses id {} already claimed by another action",
                    g.name, a.name, a.id
                ),
            });
        }
    }
    if seen_ids.contains(&0) {
        return Err(GoapDesugarError {
            span: g.span,
            message: format!(
                "goap `{}`: action id 0 is reserved for \"nothing to do\" — pick a nonzero id",
                g.name
            ),
        });
    }

    // ---- cheapest producer per fact (Step 2: the compile-time search) ----
    // For each fact produced by more than one action, keep the one with the
    // lowest `action.cost + sum(cost-to-achieve each of its own
    // requirements)`, recursively. Cycle-checked via a "currently visiting"
    // set — a fact whose cheapest path depends on itself (directly or
    // transitively) is a compile error, not a runtime hang.
    let mut plan: HashMap<&str, FactPlan> = HashMap::new();
    let mut memo_cost: HashMap<&str, f64> = HashMap::new();
    let mut visiting: HashSet<&str> = HashSet::new();
    for f in &g.facts {
        compute_fact_plan(
            &g.name,
            f.name.as_str(),
            &g.actions,
            &mut plan,
            &mut memo_cost,
            &mut visiting,
            span,
        )?;
    }

    // ---- Step 3: build the decision expression ----
    let chosen = build_requires_chain(&g.goal.requires, &fact_exprs, &plan, 0, span)?;

    let mut body: Vec<Stmt> = Vec::new();
    body.push(Stmt::Let { name: "chosen".to_string(), value: chosen, span });
    body.push(Stmt::Expr(Expr {
        kind: ExprKind::Call(
            Box::new(Expr {
                kind: ExprKind::Field(
                    Box::new(Expr { kind: ExprKind::Ident("agents".to_string()), span }),
                    format!("set_{}", g.output),
                ),
                span,
            }),
            vec![
                CallArg { name: None, value: Expr { kind: ExprKind::Ident("self".to_string()), span }, span },
                CallArg { name: None, value: Expr { kind: ExprKind::Ident("chosen".to_string()), span }, span },
            ],
        ),
        span,
    }));

    Ok(PhysicsDecl {
        annotations: vec![Annotation {
            name: "phase".to_string(),
            args: vec![AnnotationArg {
                key: None,
                value: AnnotationValue::Ident("per_agent".to_string()),
                span,
            }],
            span,
        }],
        name: g.name.clone(),
        params: Vec::new(),
        handlers: vec![PhysicsHandler {
            pattern: PhysicsPattern::Kind(EventPattern {
                name: "Tick".to_string(),
                bindings: Vec::new(),
                span,
            }),
            where_clause: None,
            body,
            span,
        }],
        cpu_only: false,
        span,
    })
}

/// Recursively resolve fact `name`'s cheapest producer (if any), memoizing
/// both the chosen producer (`plan`) and its total cost (`memo_cost`) so a
/// fact required by several actions is only searched once. `visiting`
/// detects a cycle: a fact reachable only through itself is a compile
/// error, never a silent infinite recursion.
fn compute_fact_plan<'a>(
    goap_name: &str,
    name: &'a str,
    actions: &'a [GoapActionDecl],
    plan: &mut HashMap<&'a str, FactPlan<'a>>,
    memo_cost: &mut HashMap<&'a str, f64>,
    visiting: &mut HashSet<&'a str>,
    goap_span: Span,
) -> Result<f64, GoapDesugarError> {
    if let Some(&c) = memo_cost.get(name) {
        return Ok(c);
    }
    if !visiting.insert(name) {
        return Err(GoapDesugarError {
            span: goap_span,
            message: format!(
                "goap `{goap_name}`: cyclic action dependency involving fact `{name}` — an \
                 action can never (even transitively) require a fact it also produces"
            ),
        });
    }

    let producers: Vec<&GoapActionDecl> =
        actions.iter().filter(|a| a.produces.iter().any(|p| p == name)).collect();

    let result = if producers.is_empty() {
        plan.insert(name, FactPlan { producer: None });
        Ok(0.0)
    } else {
        let mut best: Option<(&GoapActionDecl, f64)> = None;
        for a in producers {
            let mut total = a.cost;
            for req in &a.requires {
                total += compute_fact_plan(goap_name, req, actions, plan, memo_cost, visiting, goap_span)?;
            }
            if best.map(|(_, c)| total < c).unwrap_or(true) {
                best = Some((a, total));
            }
        }
        let (action, cost) = best.expect("producers is non-empty");
        plan.insert(name, FactPlan { producer: Some(action) });
        Ok(cost)
    };

    visiting.remove(name);
    if let Ok(c) = result {
        memo_cost.insert(name, c);
    }
    result
}

/// `requires` in declaration order: "if the first is already true, check
/// the rest; if not, go work on satisfying the first." `fallback` is the
/// literal committed to when every requirement in the list already holds
/// (the calling action's own id, or `0` — "nothing to do" — at the goal's
/// top level).
fn build_requires_chain(
    requires: &[String],
    facts: &HashMap<&str, &Expr>,
    plan: &HashMap<&str, FactPlan>,
    fallback: i64,
    span: Span,
) -> Result<Expr, GoapDesugarError> {
    let Some((first, rest)) = requires.split_first() else {
        return Ok(Expr { kind: ExprKind::Int(fallback), span });
    };
    let first_expr = (*facts[first.as_str()]).clone();
    let then_expr = build_requires_chain(rest, facts, plan, fallback, span)?;
    let else_expr = build_expr_for_fact(first, facts, plan, span)?;
    Ok(Expr {
        kind: ExprKind::If {
            cond: Box::new(first_expr),
            then_expr: Box::new(then_expr),
            else_expr: Some(Box::new(else_expr)),
        },
        span,
    })
}

/// "Is fact `name` already true; if so, nothing to do FOR IT (`0`); if not,
/// and it has a producing action, go check that action's own
/// requirements (recursing into `build_requires_chain`); if not, and it
/// has no producer (a terminal/gate fact), also `0` — planning cannot help
/// a condition nothing in the graph produces."
fn build_expr_for_fact(
    name: &str,
    facts: &HashMap<&str, &Expr>,
    plan: &HashMap<&str, FactPlan>,
    span: Span,
) -> Result<Expr, GoapDesugarError> {
    let fact_expr = (*facts[name]).clone();
    let Some(action) = plan[name].producer else {
        return Ok(Expr { kind: ExprKind::Int(0), span });
    };
    let then_expr = build_requires_chain(&action.requires, facts, plan, action.id, span)?;
    Ok(Expr {
        kind: ExprKind::If {
            cond: Box::new(fact_expr),
            then_expr: Box::new(Expr { kind: ExprKind::Int(0), span }),
            else_expr: Some(Box::new(then_expr)),
        },
        span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{GoapFact, GoapGoalDecl};

    const S: Span = Span::dummy();

    fn fact(name: &str, val: bool) -> GoapFact {
        GoapFact { name: name.to_string(), expr: Expr { kind: ExprKind::Bool(val), span: S }, span: S }
    }

    fn action(name: &str, requires: &[&str], produces: &[&str], cost: f64, id: i64) -> GoapActionDecl {
        GoapActionDecl {
            name: name.to_string(),
            requires: requires.iter().map(|s| s.to_string()).collect(),
            produces: produces.iter().map(|s| s.to_string()).collect(),
            cost,
            id,
            span: S,
        }
    }

    /// Evaluate the constant-folded decision tree `desugar_one` produced —
    /// valid here because every fact in this test is a literal `Bool`, so
    /// the whole tree collapses to a single `Int`.
    fn eval_const(e: &Expr) -> i64 {
        match &e.kind {
            ExprKind::Int(n) => *n,
            ExprKind::If { cond, then_expr, else_expr } => match &cond.kind {
                ExprKind::Bool(true) => eval_const(then_expr),
                ExprKind::Bool(false) => eval_const(else_expr.as_ref().expect("else present")),
                other => panic!("expected a constant-folded Bool condition, got {other:?}"),
            },
            other => panic!("expected Int or If, got {other:?}"),
        }
    }

    /// Build the colonist-economy example (chop -> build hearth; forage;
    /// cook needs both) with `has_timber`/`has_hearth`/`has_raw_food`
    /// pinned to the scenario under test, and return the chosen action id.
    fn plan_for(has_timber: bool, has_hearth: bool, has_raw_food: bool, has_meal: bool) -> i64 {
        const CHOP: i64 = 1;
        const BUILD_HEARTH: i64 = 2;
        const FORAGE: i64 = 3;
        const COOK: i64 = 4;
        let g = GoapDecl {
            annotations: vec![],
            name: "ColonistPlan".to_string(),
            facts: vec![
                fact("has_timber", has_timber),
                fact("has_hearth", has_hearth),
                fact("has_raw_food", has_raw_food),
                fact("has_meal", has_meal),
            ],
            actions: vec![
                action("ChopWood", &[], &["has_timber"], 1.0, CHOP),
                action("BuildHearth", &["has_timber"], &["has_hearth"], 2.0, BUILD_HEARTH),
                action("Forage", &[], &["has_raw_food"], 1.0, FORAGE),
                action("Cook", &["has_hearth", "has_raw_food"], &["has_meal"], 1.5, COOK),
            ],
            goal: GoapGoalDecl { requires: vec!["has_meal".to_string()], span: S },
            output: "chosen_action".to_string(),
            span: S,
        };
        let physics = desugar_one(&g).expect("desugar succeeds");
        let Stmt::Let { value, .. } = &physics.handlers[0].body[0] else {
            panic!("expected the synthesized body's first statement to be `let chosen = ...`")
        };
        eval_const(value)
    }

    #[test]
    fn starting_from_nothing_chops_wood_first() {
        // Deepest unmet prerequisite in the whole chain: no timber, no
        // hearth, no food, no meal — the ONLY thing with zero remaining
        // requirements is ChopWood.
        assert_eq!(plan_for(false, false, false, false), 1);
    }

    #[test]
    fn timber_in_hand_moves_on_to_building_the_hearth() {
        assert_eq!(plan_for(true, false, false, false), 2);
    }

    #[test]
    fn hearth_built_but_hungry_forages_for_the_independent_branch() {
        // has_hearth is satisfied; has_raw_food is the other, INDEPENDENT
        // prerequisite Cook still needs — proves multi-prerequisite (DAG,
        // not just linear-chain) resolution picks the right branch.
        assert_eq!(plan_for(true, true, false, false), 3);
    }

    #[test]
    fn hearth_and_food_in_hand_finally_cooks() {
        assert_eq!(plan_for(true, true, true, false), 4);
    }

    #[test]
    fn goal_already_met_plans_nothing() {
        assert_eq!(plan_for(true, true, true, true), 0);
    }

    #[test]
    fn two_agents_in_different_states_choose_different_actions() {
        // The load-bearing per-agent claim: the SAME compiled graph (same
        // `GoapDecl`) yields a DIFFERENT chosen action depending only on
        // which facts are true — exactly what lets two agents running the
        // identical compiled kernel diverge on their own current state.
        let stocked = plan_for(true, false, false, false);
        let bare = plan_for(false, false, false, false);
        assert_ne!(stocked, bare);
        assert_eq!(stocked, 2); // BuildHearth
        assert_eq!(bare, 1); // ChopWood
    }

    #[test]
    fn cyclic_dependency_is_a_compile_error_not_a_hang() {
        let g = GoapDecl {
            annotations: vec![],
            name: "Cyclic".to_string(),
            facts: vec![fact("a", false), fact("b", false)],
            actions: vec![
                action("MakeA", &["b"], &["a"], 1.0, 1),
                action("MakeB", &["a"], &["b"], 1.0, 2),
            ],
            goal: GoapGoalDecl { requires: vec!["a".to_string()], span: S },
            output: "chosen".to_string(),
            span: S,
        };
        let err = desugar_one(&g).expect_err("a cycle must be rejected at compile time");
        assert!(err.message.contains("cyclic"), "message was: {}", err.message);
    }

    #[test]
    fn undeclared_fact_reference_is_a_compile_error() {
        let g = GoapDecl {
            annotations: vec![],
            name: "Typo".to_string(),
            facts: vec![fact("has_timber", false)],
            actions: vec![action("ChopWood", &[], &["has_timber"], 1.0, 1)],
            goal: GoapGoalDecl { requires: vec!["has_tmiber".to_string()], span: S },
            output: "chosen".to_string(),
            span: S,
        };
        let err = desugar_one(&g).expect_err("a typo'd fact name must be rejected");
        assert!(err.message.contains("undeclared fact"), "message was: {}", err.message);
    }
}
