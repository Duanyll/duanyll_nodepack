import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    function: { nodeColor: "#0F6B57", nodeBgColor: "#0D5949" },
    high_level: { nodeColor: "#2e3e57", nodeBgColor: "#4b5b73" },
    side_effect: { nodeColor: "#3e2e40", nodeBgColor: "#573647" },
};

const NODE_COLORS = {
    "__FunctionParam__": "function",
    "__FunctionEnd__": "function",
    "duanyll::CallClosure": "function",

    "duanyll::HighLevelMap": "high_level",
    "duanyll::HighLevelMapIndexed": "high_level",
    "duanyll::HighLevelComap": "high_level",
    "duanyll::HighLevelNest": "high_level",
    "duanyll::HighLevelNestWhile": "high_level",
    "duanyll::HighLevelFold": "high_level",
    "duanyll::HighLevelTakeWhile": "high_level",
    "duanyll::HighLevelSelect": "high_level",

    "__Sow__": "side_effect",
    "__Reap__": "side_effect",
    "__Inspect__": "side_effect",
    "duanyll::Latch": "side_effect",
    "duanyll::Sleep": "side_effect",
};

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [140, 80];
        node.size[0] = theme.width;
    }
}

const ext = {
    name: "duanyll.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);