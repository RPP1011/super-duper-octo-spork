//! `goap` blocks parse and desugar into an ordinary per-agent `physics`
//! rule before resolution ever runs — see `dsl_ast::goap`'s module doc.
//! `crates/dsl_ast/src/goap.rs`'s own unit tests prove the backward-
//! chaining ALGORITHM is correct; this file proves the SOURCE SYNTAX a
//! `.sim` author actually writes parses into that same shape.

use dsl_ast::ast::{Decl, ExprKind, Stmt};

const SRC: &str = "\
entity Colonist : Agent { pos: vec3 }

event Tick { }

field inv_timber: f32
field inv_hearth: u32
field inv_raw_food: f32
field inv_meal: f32
field chosen_action: u32

goap ColonistPlan {
  fact has_timber = self.inv_timber >= 4.0
  fact has_hearth = self.inv_hearth >= 1u
  fact has_raw_food = self.inv_raw_food >= 1.0
  fact has_meal = self.inv_meal >= 1.0

  action ChopWood {
    produces: [has_timber]
    cost: 1.0
    id: 1
  }

  action BuildHearth {
    requires: [has_timber]
    produces: [has_hearth]
    cost: 2.0
    id: 2
  }

  action Forage {
    produces: [has_raw_food]
    cost: 1.0
    id: 3
  }

  action Cook {
    requires: [has_hearth, has_raw_food]
    produces: [has_meal]
    cost: 1.5
    id: 4
  }

  goal {
    requires: [has_meal]
  }

  output chosen_action
}
";

#[test]
fn goap_block_desugars_into_a_per_agent_physics_rule() {
    let p = dsl_ast::parse(SRC).expect("goap source parses");

    // No `Decl::Goap` survives past parsing.
    assert!(!p.decls.iter().any(|d| matches!(d, Decl::Goap(_))), "goap decl should have desugared away");

    let physics = p
        .decls
        .iter()
        .find_map(|d| if let Decl::Physics(ph) = d { Some(ph) } else { None })
        .expect("desugared physics rule present");
    assert_eq!(physics.name, "ColonistPlan");
    assert!(
        physics.annotations.iter().any(|a| a.name == "phase"),
        "synthesized rule must carry @phase(per_agent)"
    );
    assert_eq!(physics.handlers.len(), 1);

    let body = &physics.handlers[0].body;
    assert_eq!(body.len(), 2, "expected `let chosen = ...;` then the output write");
    assert!(matches!(&body[0], Stmt::Let { name, .. } if name == "chosen"));
    let Stmt::Expr(write) = &body[1] else { panic!("expected the output write as a bare Expr statement") };
    let ExprKind::Call(callee, args) = &write.kind else { panic!("expected a Call") };
    let ExprKind::Field(base, method) = &callee.kind else { panic!("expected `agents.set_...`") };
    assert!(matches!(&base.kind, ExprKind::Ident(n) if n == "agents"));
    assert_eq!(method, "set_chosen_action");
    assert_eq!(args.len(), 2);
}

/// Resolution (`dsl_ast::resolve`) needs no `Goap`-specific handling at
/// all — the desugared physics rule is ordinary enough to resolve and
/// lower exactly like a hand-written one. This is the load-bearing claim
/// that the rest of the compiler pipeline (IR, CG lowering, WGSL
/// emission) required zero new code for `goap` to work end to end.
#[test]
fn desugared_rule_resolves_cleanly() {
    let program = dsl_ast::parse(SRC).expect("parses");
    let compilation = dsl_ast::resolve::resolve(program).expect("resolves with no goap-specific support");
    assert_eq!(compilation.physics.len(), 1);
    assert_eq!(compilation.physics[0].name, "ColonistPlan");
}
