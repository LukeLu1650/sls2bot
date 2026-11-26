using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

namespace NecroPilot;

internal static class SelectionControls
{
    internal static (bool IsOpen, NConfirmButton? Button) Preview(NCardGridSelectionScreen screen)
    {
        return screen switch
        {
            NDeckUpgradeSelectScreen =>
                FieldPreview(screen, "_upgradeSinglePreviewContainer", "_singlePreviewConfirmButton") is { IsOpen: true } s ? s :
                FieldPreview(screen, "_upgradeMultiPreviewContainer", "_multiPreviewConfirmButton") is { IsOpen: true } m ? m :
                FindPreview(screen, "Confirm", "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer"),

            NDeckEnchantSelectScreen =>
                FieldPreview(screen, "_enchantSinglePreviewContainer", "_singlePreviewConfirmButton") is { IsOpen: true } s ? s :
                FieldPreview(screen, "_enchantMultiPreviewContainer", "_multiPreviewConfirmButton") is { IsOpen: true } m ? m :
                FindPreview(screen, "Confirm", "%EnchantSinglePreviewContainer", "%EnchantMultiPreviewContainer"),

            NDeckTransformSelectScreen =>
                FieldPreview(screen, "_previewContainer", "_previewConfirmButton") is { IsOpen: true } p ? p :
                FindPreview(screen, "Confirm", "%PreviewContainer"),

            NDeckCardSelectScreen =>
                FieldPreview(screen, "_previewContainer", "_previewConfirmButton") is { IsOpen: true } p ? p :
                FindPreview(screen, "%PreviewConfirm", "%PreviewContainer"),

            _ => (false, null),
        };
    }

    internal static NConfirmButton? GridConfirm(NCardGridSelectionScreen screen)
    {
        if (AccessTools.Field(screen.GetType(), "_confirmButton")?.GetValue(screen) is NConfirmButton fieldBtn)
            return Ready(fieldBtn);
        string path = screen is NDeckEnchantSelectScreen or NDeckTransformSelectScreen ? "Confirm" : "%Confirm";
        return Ready(screen.GetNodeOrNull<NConfirmButton>(path));
    }

    private static (bool IsOpen, NConfirmButton? Button) FieldPreview(
        NCardGridSelectionScreen screen, string containerField, string buttonField)
    {
        var t = screen.GetType();
        if (AccessTools.Field(t, containerField)?.GetValue(screen) is not Control container || !container.IsVisibleInTree())
            return (false, null);
        var btn = AccessTools.Field(t, buttonField)?.GetValue(screen) as NConfirmButton;
        return (true, Ready(btn));
    }

    private static (bool IsOpen, NConfirmButton? Button) FindPreview(
        Node screen, string buttonPath, string containerPath, string? secondContainerPath = null)
    {
        var container = screen.GetNodeOrNull<Control>(containerPath);
        if (container?.IsVisibleInTree() != true && secondContainerPath != null)
            container = screen.GetNodeOrNull<Control>(secondContainerPath);
        if (container?.IsVisibleInTree() != true)
            return (false, null);
        return (true, Ready(container.GetNodeOrNull<NConfirmButton>(buttonPath)));
    }

    private static NConfirmButton? Ready(NConfirmButton? button) =>
        button is { IsEnabled: true } && button.IsVisibleInTree() ? button : null;
}
