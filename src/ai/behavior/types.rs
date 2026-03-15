/// AST types for the parsed behavior DSL.

#[derive(Debug, Clone)]
pub struct BehaviorTree {
    pub name: String,
    pub rules: Vec<Rule>,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub priority: RulePriority,
    pub condition: Option<Condition>,
    pub action: Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RulePriority {
    Priority,
    Default,
    Fallback,
}

#[derive(Debug, Clone)]
pub enum Action {
    Chase(Target),
    Flee(Target),
    MoveTo(Position),
    MaintainDistance(Target, f32),
    Attack(Target),
    Focus(Target),
    CastAbility(usize, Target),
    CastIfReady(usize, Target),
    UseBestAbility,
    UseBestAbilityOn(Target),
    UseAbilityType(AbilityCategory, Option<Target>),
    Hold,
    Run(String),
}

#[derive(Debug, Clone)]
pub enum Target {
    Self_,
    NearestEnemy,
    NearestAlly,
    LowestHpEnemy,
    LowestHpAlly,
    HighestDpsEnemy,
    HighestThreatEnemy,
    CastingEnemy,
    EnemyAttacking(Box<Target>),
    Tagged(String),
    UnitId(u32),
}

#[derive(Debug, Clone)]
pub enum Position {
    Entity(Target),
    Fixed(f32, f32),
    Random,
    TargetPosition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbilityCategory {
    Damage,
    Heal,
    Cc,
    Buff,
    Aoe,
}

#[derive(Debug, Clone)]
pub enum Condition {
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Not(Box<Condition>),
    Compare(Value, CompOp, Value),
    StateCheck(StateCheck),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    Lt,
    Gt,
    Lte,
    Gte,
    Eq,
    Neq,
}

#[derive(Debug, Clone)]
pub enum Value {
    Number(f32),
    SelfHp,
    SelfHpPct,
    TargetHp(Target),
    TargetHpPct(Target),
    TargetDistance(Target),
    TargetDps(Target),
    TargetCcRemaining(Target),
    TargetCastProgress(Target),
    EnemyCount,
    AllyCount,
    EnemyCountInRange(f32),
    AllyCountBelowHp(f32),
    Tick,
    AbilityCooldownPct(usize),
    BestAbilityUrgency,
}

#[derive(Debug, Clone)]
pub enum StateCheck {
    HealReady,
    CcReady,
    AoeReady,
    AbilityReady(usize),
    IsCasting,
    IsCcd,
    TargetIsCasting(Target),
    TargetIsCcd(Target),
    CanAttack,
    InDangerZone,
    AllyInDanger,
    NearWall(f32),
    Every(u64),
}
